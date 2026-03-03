# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
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
        preserve_length (bool): If True, use padding so output T equals input T
            (for CNN+Transformer hybrid). If False, output is shorter (original
            TDS behavior). (default: False)
    """

    def __init__(
        self,
        channels: int,
        width: int,
        kernel_width: int,
        preserve_length: bool = False,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.width = width
        self.preserve_length = preserve_length
        self.kernel_width = kernel_width

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        if self.preserve_length:
            # Asymmetric padding for exact length preservation (odd kernel_width
            # e.g. 32 requires pad_left=15, pad_right=16 for output T = input T)
            pad_l = (self.kernel_width - 1) // 2
            pad_r = self.kernel_width - 1 - pad_l
            x = nn.functional.pad(x, (pad_l, pad_r), mode="constant", value=0)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

        if self.preserve_length:
            x = x + inputs
        else:
            T_out = x.shape[0]
            x = x + inputs[-T_out:]

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


class CNNTransformerEncoder(nn.Module):
    """CNN + Transformer hybrid encoder. Stacks length-preserving TDS conv blocks
    (local feature extraction) followed by a transformer (global context).
    No temporal downsampling — output length equals input length for CTC.

    Args:
        num_features (int): Feature dimension (d_model).
        cnn_block_channels (list): Channel config for each CNN block.
        cnn_kernel_width (int): Temporal kernel size for CNN blocks.
        nhead (int): Number of transformer attention heads.
        num_layers (int): Number of transformer encoder layers.
        dim_feedforward (int | None): FFN hidden dim. (default: 4 * d_model)
        dropout (float): Dropout probability. (default: 0.1)
    """

    def __init__(
        self,
        num_features: int,
        cnn_block_channels: Sequence[int],
        cnn_kernel_width: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        cnn_blocks: list[nn.Module] = []
        for channels in cnn_block_channels:
            assert num_features % channels == 0
            width = num_features // channels
            cnn_blocks.extend(
                [
                    TDSConv2dBlock(
                        channels, width, cnn_kernel_width, preserve_length=True
                    ),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.cnn = nn.Sequential(*cnn_blocks)
        self.transformer = TransformerEncoderStack(
            d_model=num_features,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.cnn(x)
        return self.transformer(x, input_lengths=input_lengths)


# -----------------------------------------------------------------------------
# Transformer encoder for CTC (replaces TDSConvEncoder)
# -----------------------------------------------------------------------------


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding. Adds positional information to sequences.

    Args:
        d_model (int): Embedding dimension.
        dropout (float): Dropout probability. (default: 0.1)
        max_len (int): Maximum sequence length for precomputed buffer. Sequences
            longer than this are supported via on-the-fly computation. (default: 5000)
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def _get_pe(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Return positional encoding of shape (seq_len, 1, d_model)."""
        if seq_len <= self.max_len:
            return self.pe[:seq_len]
        # On-the-fly compute for sequences longer than precomputed buffer
        # (e.g. test-time with full sessions, window_length=None)
        pe = torch.zeros(seq_len, self.d_model, device=device, dtype=self.pe.dtype)
        position = torch.arange(0, seq_len, device=device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(1)  # (seq_len, 1, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (T, N, d_model)"""
        x = x + self._get_pe(x.size(0), x.device)
        return self.dropout(x)


class TransformerEncoderStack(nn.Module):
    """Transformer encoder stack for sequence-to-sequence with CTC.

    Uses the same input/output format as TDSConvEncoder: (T, N, num_features).
    No temporal downsampling — output length equals input length for CTC.

    Args:
        d_model (int): Model dimension (must match num_features from frontend).
        nhead (int): Number of attention heads.
        num_layers (int): Number of transformer encoder layers.
        dim_feedforward (int): FFN hidden dimension. (default: 4 * d_model)
        dropout (float): Dropout probability. (default: 0.1)
        activation (str): FFN activation. (default: "gelu")
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        dim_feedforward = dim_feedforward or 4 * d_model

        self.pos_encoder = SinusoidalPositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=False,  # (T, N, D) format
            norm_first=True,  # Pre-norm for more stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,  # Ensure consistent behavior with padding
        )

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (T, N, d_model) input features.
            input_lengths: (N,) actual lengths for each batch item. Used to build
                key_padding_mask so padding positions are ignored. If None, no mask.

        Returns:
            (T, N, d_model) output.
        """
        x = self.pos_encoder(x)
        key_padding_mask = None
        if input_lengths is not None:
            # key_padding_mask: (N, T), True = ignore (padding)
            T, N = x.shape[0], x.shape[1]
            key_padding_mask = torch.arange(T, device=x.device)[None, :] >= input_lengths[:, None]
        return self.transformer(x, src_key_padding_mask=key_padding_mask)
