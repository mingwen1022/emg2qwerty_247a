# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics import MetricCollection

from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.modules import (
    CNNTransformerEncoder,
<<<<<<< HEAD
    GRUEncoder,
    MultiBandRotationInvariantMLP,
    RawEMGCNNEncoder,
    SpectrogramNorm,
    TDSConvEncoder,
    TDSFullyConnectedBlock,
    TransformerEncoderStack,
    TwoLayerFCBlock,
    VanillaRNNEncoder,
=======
    MultiBandRotationInvariantMLP,
    SpectrogramNorm,
    TDSConvEncoder,
    TransformerEncoderStack,
>>>>>>> 308ab0a (SH)
)
from emg2qwerty.transforms import Transform


class WindowedEMGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        train_transform: Transform[np.ndarray, torch.Tensor],
        val_transform: Transform[np.ndarray, torch.Tensor],
        test_transform: Transform[np.ndarray, torch.Tensor],
<<<<<<< HEAD
        stride: int | None = None,
=======
>>>>>>> 308ab0a (SH)
        test_window_length: int | None = None,
    ) -> None:
        super().__init__()

        self.window_length = window_length
<<<<<<< HEAD
        self.stride = stride
=======
>>>>>>> 308ab0a (SH)
        self.padding = padding

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
<<<<<<< HEAD
=======

>>>>>>> 308ab0a (SH)
        # If None, feed entire sessions at test (original behavior). If set (e.g.
        # to window_length), use windowed test for fair comparison with val.
        # Transformers require windowed test since they're trained on short seqs.
        self.test_window_length = test_window_length

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.train_transform,
                    window_length=self.window_length,
<<<<<<< HEAD
                    stride=self.stride,
=======
>>>>>>> 308ab0a (SH)
                    padding=self.padding,
                    jitter=True,
                )
                for hdf5_path in self.train_sessions
            ]
        )
        self.val_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.val_transform,
                    window_length=self.window_length,
<<<<<<< HEAD
                    stride=self.stride,
=======
>>>>>>> 308ab0a (SH)
                    padding=self.padding,
                    jitter=False,
                )
                for hdf5_path in self.val_sessions
            ]
        )
        test_wlen = self.test_window_length
        test_pad = self.padding if test_wlen is not None else (0, 0)
        self.test_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.test_transform,
                    window_length=test_wlen if test_wlen is not None else None,
                    padding=test_pad,
                    jitter=False,
                )
                for hdf5_path in self.test_sessions
            ]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        # When test_window_length is None, entire sessions are fed at once;
        # use batch_size=1 to fit in GPU memory. When windowed, use batch_size.
        batch_size = 1 if self.test_window_length is None else self.batch_size
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )


class TDSConvCTCModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
        electrode_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.electrode_channels = electrode_channels or self.ELECTRODE_CHANNELS
<<<<<<< HEAD
        self.save_hyperparameters(
            ignore=["optimizer", "lr_scheduler", "decoder", "mlp_features", "block_channels"]
        )
        self._optimizer_config = optimizer
        self._lr_scheduler_config = lr_scheduler
=======
        self.save_hyperparameters()
>>>>>>> 308ab0a (SH)

        num_features = self.NUM_BANDS * mlp_features[-1]

        # Model
        # inputs: (T, N, bands=2, electrode_channels, freq)
        self.model = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.electrode_channels),
            # (T, N, bands=2, mlp_features[-1])
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            # (T, N, num_features)
            nn.Flatten(start_dim=2),
            TDSConvEncoder(
                num_features=num_features,
                block_channels=block_channels,
                kernel_width=kernel_width,
            ),
            # (T, N, num_classes)
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)

        # Shrink input lengths by an amount equivalent to the conv encoder's
        # temporal receptive field to compute output activation lengths for CTCLoss.
        # NOTE: This assumes the encoder doesn't perform any temporal downsampling
        # such as by striding.
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Unpad targets (T, N) for batch entry
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
<<<<<<< HEAD
            optimizer_config=self._optimizer_config,
            lr_scheduler_config=self._lr_scheduler_config,
=======
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
>>>>>>> 308ab0a (SH)
        )


class TransformerCTCModule(pl.LightningModule):
    """Transformer encoder + CTC for EMG-to-text. Uses the same frontend as
    TDSConvCTCModule (SpectrogramNorm + MultiBandRotationInvariantMLP) and
    replaces the TDSConvEncoder with a TransformerEncoderStack.

    No temporal downsampling — emission_lengths = input_lengths for CTC.
    """

    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        d_model: int,
        nhead: int,
        num_layers: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        electrode_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.electrode_channels = electrode_channels or self.ELECTRODE_CHANNELS
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]
        assert d_model == num_features, (
            f"d_model ({d_model}) must match num_features ({num_features}) "
            "from frontend (NUM_BANDS * mlp_features[-1])"
        )

        self.spectrogram_norm = SpectrogramNorm(
            channels=self.NUM_BANDS * self.electrode_channels
        )
        self.mlp = MultiBandRotationInvariantMLP(
            in_features=in_features,
            mlp_features=mlp_features,
            num_bands=self.NUM_BANDS,
        )
        self.transformer = TransformerEncoderStack(
            d_model=num_features,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.output_proj = nn.Linear(num_features, charset().num_classes)

        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self.decoder = instantiate(decoder)

        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.spectrogram_norm(inputs)
        x = self.mlp(x)
        x = x.flatten(start_dim=2)
        x = self.transformer(x, input_lengths=input_lengths)
        return self.output_proj(x).log_softmax(dim=-1)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)

        emissions = self.forward(inputs, input_lengths=input_lengths)

        # No temporal downsampling: emission_lengths = input_lengths
        emission_lengths = input_lengths

        loss = self.ctc_loss(
            log_probs=emissions,
            targets=targets.transpose(0, 1),
            input_lengths=emission_lengths,
            target_lengths=target_lengths,
        )

        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        metrics = self.metrics[f"{phase}_metrics"]
        targets_np = targets.detach().cpu().numpy()
        target_lengths_np = target_lengths.detach().cpu().numpy()
        for i in range(N):
            target = LabelData.from_labels(targets_np[: target_lengths_np[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )


<<<<<<< HEAD
class RNNCTCModule(pl.LightningModule):
    """Vanilla RNN encoder + CTC. Uses same frontend as TDS, replaces conv with RNN."""

    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        rnn_hidden_size: int,
        rnn_num_layers: int,
        rnn_bidirectional: bool,
        rnn_nonlinearity: str,
        rnn_dropout: float,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=["optimizer", "lr_scheduler", "decoder", "mlp_features"]
        )
        self._optimizer_config = optimizer
        self._lr_scheduler_config = lr_scheduler

        num_features = self.NUM_BANDS * mlp_features[-1]
        self.frontend = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),
        )
        self.encoder = VanillaRNNEncoder(
            input_size=num_features,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            bidirectional=rnn_bidirectional,
            nonlinearity=rnn_nonlinearity,
            dropout=rnn_dropout,
        )
        self.classifier = nn.Sequential(
            TwoLayerFCBlock(self.encoder.output_size),
            nn.Linear(self.encoder.output_size, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self.decoder = instantiate(decoder)

        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.frontend(inputs)
        x = self.encoder(x)
        return self.classifier(x)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)

        emissions = self.forward(inputs)
        emission_lengths = input_lengths

        loss = self.ctc_loss(
            log_probs=emissions,
            targets=targets.transpose(0, 1),
            input_lengths=emission_lengths,
            target_lengths=target_lengths,
        )

        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        metrics = self.metrics[f"{phase}_metrics"]
        targets_np = targets.detach().cpu().numpy()
        target_lengths_np = target_lengths.detach().cpu().numpy()
        for i in range(N):
            target = LabelData.from_labels(targets_np[: target_lengths_np[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self._optimizer_config,
            lr_scheduler_config=self._lr_scheduler_config,
        )


=======
>>>>>>> 308ab0a (SH)
class CNNTransformerCTCModule(pl.LightningModule):
    """CNN + Transformer hybrid + CTC for EMG-to-text. Uses the same frontend as
    TDSConvCTCModule and TransformerCTCModule. Stacks length-preserving CNN blocks
    (local feature extraction) followed by a transformer (global context).

    No temporal downsampling — emission_lengths = input_lengths for CTC.
    """

    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        d_model: int,
        cnn_block_channels: Sequence[int],
        cnn_kernel_width: int,
        nhead: int,
        num_layers: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        electrode_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.electrode_channels = electrode_channels or self.ELECTRODE_CHANNELS
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]
        assert d_model == num_features, (
            f"d_model ({d_model}) must match num_features ({num_features}) "
            "from frontend (NUM_BANDS * mlp_features[-1])"
        )

        self.spectrogram_norm = SpectrogramNorm(
            channels=self.NUM_BANDS * self.electrode_channels
        )
        self.mlp = MultiBandRotationInvariantMLP(
            in_features=in_features,
            mlp_features=mlp_features,
            num_bands=self.NUM_BANDS,
        )
        self.encoder = CNNTransformerEncoder(
            num_features=num_features,
            cnn_block_channels=cnn_block_channels,
            cnn_kernel_width=cnn_kernel_width,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.output_proj = nn.Linear(num_features, charset().num_classes)

        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self.decoder = instantiate(decoder)

        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.spectrogram_norm(inputs)
        x = self.mlp(x)
        x = x.flatten(start_dim=2)
        x = self.encoder(x, input_lengths=input_lengths)
        return self.output_proj(x).log_softmax(dim=-1)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)

        emissions = self.forward(inputs, input_lengths=input_lengths)
        emission_lengths = input_lengths

        loss = self.ctc_loss(
            log_probs=emissions,
            targets=targets.transpose(0, 1),
            input_lengths=emission_lengths,
            target_lengths=target_lengths,
        )

        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        metrics = self.metrics[f"{phase}_metrics"]
        targets_np = targets.detach().cpu().numpy()
        target_lengths_np = target_lengths.detach().cpu().numpy()
        for i in range(N):
            target = LabelData.from_labels(targets_np[: target_lengths_np[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )
<<<<<<< HEAD


class GRUCTCModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        gru_hidden_size: int,
        gru_num_layers: int,
        gru_bidirectional: bool,
        gru_dropout: float,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=["optimizer", "lr_scheduler", "decoder", "mlp_features"]
        )
        self._log_hyperparams = (
            False  # TB/tensorboardX choke on any non-scalar in hparams
        )
        self._optimizer_config = optimizer
        self._lr_scheduler_config = lr_scheduler

        num_features = self.NUM_BANDS * mlp_features[-1]
        self.frontend = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),
        )
        self.encoder = GRUEncoder(
            input_size=num_features,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            bidirectional=gru_bidirectional,
            dropout=gru_dropout,
        )
        self.classifier = nn.Sequential(
            # TwoLayerFCBlock(self.encoder.output_size),
            nn.Linear(self.encoder.output_size, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self.decoder = instantiate(decoder)

        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.frontend(inputs)
        x = self.encoder(x)
        return self.classifier(x)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)
        emission_lengths = input_lengths

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self._optimizer_config,
            lr_scheduler_config=self._lr_scheduler_config,
        )


class RawCnnGruCtcModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        cnn_channels: Sequence[int],
        cnn_kernel_sizes: Sequence[int],
        cnn_strides: Sequence[int],
        cnn_paddings: Sequence[int] | None,
        cnn_dilations: Sequence[int] | None,
        cnn_use_batch_norm: bool,
        gru_hidden_size: int,
        gru_num_layers: int,
        gru_bidirectional: bool,
        gru_dropout: float,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "optimizer",
                "lr_scheduler",
                "decoder",
                "cnn_channels",
                "cnn_kernel_sizes",
                "cnn_strides",
                "cnn_paddings",
                "cnn_dilations",
            ]
        )
        self._log_hyperparams = (
            False  # TB/tensorboardX choke on any non-scalar in hparams
        )
        self._optimizer_config = optimizer
        self._lr_scheduler_config = lr_scheduler

        input_channels = self.NUM_BANDS * self.ELECTRODE_CHANNELS
        self.cnn_encoder = RawEMGCNNEncoder(
            in_channels=input_channels,
            channels=cnn_channels,
            kernel_sizes=cnn_kernel_sizes,
            strides=cnn_strides,
            paddings=cnn_paddings,
            dilations=cnn_dilations,
            use_batch_norm=cnn_use_batch_norm,
        )
        self.gru_encoder = GRUEncoder(
            input_size=self.cnn_encoder.output_size,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            bidirectional=gru_bidirectional,
            dropout=gru_dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.gru_encoder.output_size, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self.decoder = instantiate(decoder)

        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 4:
            # (T, N, bands=2, channels=16) -> (T, N, C=32)
            x = inputs.flatten(start_dim=2)
        elif inputs.ndim == 3:
            x = inputs
        else:
            raise ValueError(
                "Expected raw inputs with shape (T, N, bands, channels) "
                f"or (T, N, C), got {tuple(inputs.shape)}."
            )
        x = self.cnn_encoder(x)
        x = self.gru_encoder(x)
        return self.classifier(x)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)
        emission_lengths = self.cnn_encoder.output_lengths(input_lengths)
        if torch.any(emission_lengths <= 0).item():
            raise ValueError(
                "emission_lengths must be > 0 after CNN downsampling. "
                f"min_input_length={int(input_lengths.min().item())}, "
                f"min_emission_length={int(emission_lengths.min().item())}."
            )
        if torch.any(emission_lengths < target_lengths).item():
            raise ValueError(
                "emission_lengths must be >= target_lengths for CTC. "
                f"min_emission_length={int(emission_lengths.min().item())}, "
                f"max_target_length={int(target_lengths.max().item())}."
            )
        if torch.any(emission_lengths > emissions.shape[0]).item():
            raise ValueError(
                "emission_lengths cannot exceed emissions timesteps. "
                f"max_emission_length={int(emission_lengths.max().item())}, "
                f"emissions.T={emissions.shape[0]}."
            )

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self._optimizer_config,
            lr_scheduler_config=self._lr_scheduler_config,
        )


class CRNNCTCModule(pl.LightningModule):
    """CNN + RNN (CRNN) encoder + CTC. Uses TDS conv then Vanilla RNN."""

    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        cnn_block_channels: Sequence[int],
        cnn_kernel_width: int,
        rnn_hidden_size: int,
        rnn_num_layers: int,
        rnn_bidirectional: bool,
        rnn_nonlinearity: str,
        rnn_dropout: float,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "optimizer",
                "lr_scheduler",
                "decoder",
                "mlp_features",
                "cnn_block_channels",
            ]
        )
        self._log_hyperparams = (
            False  # TB/tensorboardX choke on any non-scalar in hparams
        )
        self._optimizer_config = optimizer
        self._lr_scheduler_config = lr_scheduler

        num_features = self.NUM_BANDS * mlp_features[-1]
        self.frontend = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),
        )
        self.cnn_encoder = TDSConvEncoder(
            num_features=num_features,
            block_channels=cnn_block_channels,
            kernel_width=cnn_kernel_width,
        )
        self.rnn_encoder = VanillaRNNEncoder(
            input_size=num_features,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            bidirectional=rnn_bidirectional,
            nonlinearity=rnn_nonlinearity,
            dropout=rnn_dropout,
        )
        self.classifier = nn.Sequential(
            TDSFullyConnectedBlock(self.rnn_encoder.output_size),
            nn.Linear(self.rnn_encoder.output_size, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self.decoder = instantiate(decoder)

        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.frontend(inputs)
        x = self.cnn_encoder(x)
        x = self.rnn_encoder(x)
        return self.classifier(x)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)

        emissions = self.forward(inputs)
        T_diff = inputs.shape[0] - emissions.shape[0]
        if T_diff < 0:
            raise ValueError(
                f"Expected non-negative T_diff, got {T_diff}. "
                f"inputs.T={inputs.shape[0]}, emissions.T={emissions.shape[0]}."
            )

        emission_lengths = input_lengths - T_diff
        if torch.any(emission_lengths <= 0).item():
            raise ValueError(
                "emission_lengths must be > 0 after CNN shrinkage. "
                f"T_diff={T_diff}, min_input_length={int(input_lengths.min().item())}, "
                f"min_emission_length={int(emission_lengths.min().item())}."
            )
        if torch.any(emission_lengths < target_lengths).item():
            raise ValueError(
                "emission_lengths must be >= target_lengths for CTC. "
                f"min_emission_length={int(emission_lengths.min().item())}, "
                f"max_target_length={int(target_lengths.max().item())}."
            )

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,
        )

        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        metrics = self.metrics[f"{phase}_metrics"]
        targets_np = targets.detach().cpu().numpy()
        target_lengths_np = target_lengths.detach().cpu().numpy()
        for i in range(N):
            target = LabelData.from_labels(targets_np[: target_lengths_np[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self._optimizer_config,
            lr_scheduler_config=self._lr_scheduler_config,
        )


class CGRUCTCModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        cnn_block_channels: Sequence[int],
        cnn_kernel_width: int,
        gru_hidden_size: int,
        gru_num_layers: int,
        gru_bidirectional: bool,
        gru_dropout: float,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "optimizer",
                "lr_scheduler",
                "decoder",
                "mlp_features",
                "cnn_block_channels",
            ]
        )
        self._log_hyperparams = (
            False  # TB/tensorboardX choke on any non-scalar in hparams
        )
        self._optimizer_config = optimizer
        self._lr_scheduler_config = lr_scheduler

        num_features = self.NUM_BANDS * mlp_features[-1]
        self.frontend = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),
        )
        self.cnn_encoder = TDSConvEncoder(
            num_features=num_features,
            block_channels=cnn_block_channels,
            kernel_width=cnn_kernel_width,
        )
        self.gru_encoder = GRUEncoder(
            input_size=num_features,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            bidirectional=gru_bidirectional,
            dropout=gru_dropout,
        )
        self.classifier = nn.Sequential(
            # TDSFullyConnectedBlock(self.gru_encoder.output_size),
            nn.Linear(self.gru_encoder.output_size, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self.decoder = instantiate(decoder)

        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.frontend(inputs)
        x = self.cnn_encoder(x)
        x = self.gru_encoder(x)
        return self.classifier(x)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)
        T_diff = inputs.shape[0] - emissions.shape[0]
        if T_diff < 0:
            raise ValueError(
                f"Expected non-negative T_diff, got {T_diff}. "
                f"inputs.T={inputs.shape[0]}, emissions.T={emissions.shape[0]}."
            )

        emission_lengths = input_lengths - T_diff
        if torch.any(emission_lengths <= 0).item():
            raise ValueError(
                "emission_lengths must be > 0 after CNN shrinkage. "
                f"T_diff={T_diff}, min_input_length={int(input_lengths.min().item())}, "
                f"min_emission_length={int(emission_lengths.min().item())}."
            )
        if torch.any(emission_lengths < target_lengths).item():
            raise ValueError(
                "emission_lengths must be >= target_lengths for CTC. "
                f"min_emission_length={int(emission_lengths.min().item())}, "
                f"max_target_length={int(target_lengths.max().item())}."
            )

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self._optimizer_config,
            lr_scheduler_config=self._lr_scheduler_config,
        )
=======
>>>>>>> 308ab0a (SH)
