# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

import torch
from torch import nn


class SpectrogramNorm(nn.Module):
    """A `torch.nn.Module` that applies 2D batch normalization over spectrogram
    per electrode channel per band. Inputs must be of shape
    (T, N, num_bands, electrode_channels, frequency_bins).

    With left and right bands and 16 electrode channels per band, spectrograms
    corresponding to each of the 2 * 16 = 32 channels are normalized
    independently using `nn.BatchNorm2d` such that stats are computed
    over (N, freq, time) slices.

    Args:
        channels (int): Total number of electrode channels across bands
            such that the normalization statistics are calculated per channel.
            Should be equal to num_bands * electrode_chanels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape  # (T, N, bands=2, C=16, freq)
        assert self.channels == bands * C

        x = inputs.movedim(0, -1)  # (N, bands=2, C=16, freq, T)
        x = x.reshape(N, bands * C, freq, T)
        x = self.batch_norm(x)
        x = x.reshape(N, bands, C, freq, T)
        return x.movedim(-1, 0)  # (T, N, bands=2, C=16, freq)


class RotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that takes an input tensor of shape
    (T, N, electrode_channels, ...) corresponding to a single band, applies
    an MLP after shifting/rotating the electrodes for each positional offset
    in ``offsets``, and pools over all the outputs.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input of
            shape (T, N, C, ...), this should be equal to C * ... (that is,
            the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                ]
            )
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)

        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, C, ...)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, C, ...) -> (T, N, rotation, C, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, C, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)


class MultiBandRotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that applies a separate instance of
    `RotationInvariantMLP` per band for inputs of shape
    (T, N, num_bands, electrode_channels, ...).

    Returns a tensor of shape (T, N, num_bands, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input
            of shape (T, N, num_bands, C, ...), this should be equal to
            C * ... (that is, the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
        num_bands (int): ``num_bands`` for an input of shape
            (T, N, num_bands, C, ...). (default: 2)
        stack_dim (int): The dimension along which the left and right data
            are stacked. (default: 2)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim

        # One MLP per band
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        outputs_per_band = [
            mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)
        ]
        return torch.stack(outputs_per_band, dim=self.stack_dim)


class VanillaRNNEncoder(nn.Module):
    """A vanilla RNN encoder over time for input tensors of shape (T, N, C).

    Args:
        input_size (int): Input feature size C.
        hidden_size (int): Hidden size per direction.
        num_layers (int): Number of stacked RNN layers.
        bidirectional (bool): Whether to use a bidirectional RNN.
        nonlinearity (str): RNN nonlinearity, "tanh" or "relu".
        dropout (float): Dropout between RNN layers. Ignored when num_layers=1.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool,
        nonlinearity: str = "tanh",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if nonlinearity not in {"tanh", "relu"}:
            raise ValueError(
                f"Unsupported nonlinearity: {nonlinearity}. Use 'tanh' or 'relu'."
            )
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}.")
        if dropout < 0.0:
            raise ValueError(f"dropout must be >= 0.0, got {dropout}.")

        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        rnn_dropout = 0.0 if num_layers == 1 else dropout
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            bidirectional=bidirectional,
            dropout=rnn_dropout,
        )

    @property
    def output_size(self) -> int:
        return self.hidden_size * (2 if self.bidirectional else 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.rnn(inputs)
        return outputs


class GRUEncoder(nn.Module):
    """A GRU encoder over time for input tensors of shape (T, N, C).

    Args:
        input_size (int): Input feature size C.
        hidden_size (int): Hidden size per direction.
        num_layers (int): Number of stacked GRU layers.
        bidirectional (bool): Whether to use a bidirectional GRU.
        dropout (float): Dropout between GRU layers. Ignored when num_layers=1.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}.")
        if dropout < 0.0:
            raise ValueError(f"dropout must be >= 0.0, got {dropout}.")

        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        gru_dropout = 0.0 if num_layers == 1 else dropout
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=gru_dropout,
        )

    @property
    def output_size(self) -> int:
        return self.hidden_size * (2 if self.bidirectional else 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.gru(inputs)
        return outputs


class RawEMGCNNEncoder(nn.Module):
    """1D CNN encoder for raw EMG inputs of shape (T, N, C).

    The module applies a stack of temporal Conv1d layers while preserving
    TNC convention for the public interface.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        strides: Sequence[int],
        paddings: Sequence[int] | None = None,
        dilations: Sequence[int] | None = None,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        num_layers = len(channels)
        if num_layers == 0:
            raise ValueError("channels must contain at least one layer.")
        if len(kernel_sizes) != num_layers:
            raise ValueError(
                "kernel_sizes must have the same length as channels. "
                f"Got {len(kernel_sizes)} and {num_layers}."
            )
        if len(strides) != num_layers:
            raise ValueError(
                "strides must have the same length as channels. "
                f"Got {len(strides)} and {num_layers}."
            )

        if paddings is None:
            paddings = tuple(0 for _ in range(num_layers))
        if dilations is None:
            dilations = tuple(1 for _ in range(num_layers))

        if len(paddings) != num_layers:
            raise ValueError(
                "paddings must have the same length as channels. "
                f"Got {len(paddings)} and {num_layers}."
            )
        if len(dilations) != num_layers:
            raise ValueError(
                "dilations must have the same length as channels. "
                f"Got {len(dilations)} and {num_layers}."
            )

        self.kernel_sizes = tuple(int(k) for k in kernel_sizes)
        self.strides = tuple(int(s) for s in strides)
        self.paddings = tuple(int(p) for p in paddings)
        self.dilations = tuple(int(d) for d in dilations)

        layers: list[nn.Module] = []
        current_in = in_channels
        for out_channels, kernel_size, stride, padding, dilation in zip(
            channels,
            self.kernel_sizes,
            self.strides,
            self.paddings,
            self.dilations,
        ):
            if kernel_size < 1:
                raise ValueError(f"kernel_size must be >= 1, got {kernel_size}.")
            if stride < 1:
                raise ValueError(f"stride must be >= 1, got {stride}.")
            if dilation < 1:
                raise ValueError(f"dilation must be >= 1, got {dilation}.")
            if padding < 0:
                raise ValueError(f"padding must be >= 0, got {padding}.")

            layers.append(
                nn.Conv1d(
                    in_channels=current_in,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                )
            )
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            current_in = out_channels
        self.network = nn.Sequential(*layers)
        self.output_size = current_in

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.permute(1, 2, 0)  # (T, N, C) -> (N, C, T)
        x = self.network(x)
        return x.permute(2, 0, 1)  # (N, C, T) -> (T, N, C)

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        lengths = input_lengths.clone()
        for kernel_size, stride, padding, dilation in zip(
            self.kernel_sizes,
            self.strides,
            self.paddings,
            self.dilations,
        ):
            lengths = (
                lengths + 2 * padding - dilation * (kernel_size - 1) - 1
            ) // stride + 1
        return lengths


class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # TNC -> NCT -> NcwT
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

        # Skip connection after downsampling
        T_out = x.shape[0]
        x = x + inputs[-T_out:]

        # Layer norm over C
        return self.layer_norm(x)  # TNC


class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # TNC
        x = self.fc_block(x)
        x = x + inputs
        return self.layer_norm(x)  # TNC


class TwoLayerFCBlock(nn.Module):
    """Two-layer FC with residual and LayerNorm (TDS-style)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.fc(x))


class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, num_features // channels, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)
