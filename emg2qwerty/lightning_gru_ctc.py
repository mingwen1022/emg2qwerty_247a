import torch
import torch.nn as nn
import pytorch_lightning as pl

from emg2qwerty.models.gru_ctc import GRUCTCNet
from emg2qwerty.decoder import compute_cer  # if available
from emg2qwerty.decoder import CTCGreedyDecoder  # baseline decoder


class GRUCTCModule(pl.LightningModule):
    """
    A minimal LightningModule that mirrors the baseline training loop:
    - forward -> logits for CTC
    - CTCLoss
    - logs CER on val/test using greedy decode
    """

    def __init__(
        self,
        in_features: int = 528,
        hidden_size: int = 384,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.0,
        fc_features=(384, 384),
        # Most repos expose num_classes via the dataset tokenizer; we’ll set later in setup()
        num_classes: int = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.net = None  # created once we know vocab size

        self.ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
        self.decoder = CTCGreedyDecoder()

    def setup(self, stage=None):
        # Try to infer num_classes from decoder / tokenizer used in repo
        # Many implementations store it on the decoder as `num_classes` or `vocab_size`.
        if self.net is None:
            nc = self.num_classes
            if nc is None:
                for attr in ["num_classes", "vocab_size"]:
                    if hasattr(self.decoder, attr):
                        nc = int(getattr(self.decoder, attr))
                        break
            if nc is None:
                # Fall back: common emg2qwerty char vocab is small; but we *shouldn’t guess*.
                raise ValueError(
                    "Could not infer num_classes. Set module.num_classes in config/model/gru_ctc.yaml "
                    "to match your repo's label vocabulary size."
                )
            self.num_classes = nc
            self.net = GRUCTCNet(
                in_features=self.hparams.in_features,
                num_classes=self.num_classes,
                hidden_size=self.hparams.hidden_size,
                num_layers=self.hparams.num_layers,
                bidirectional=self.hparams.bidirectional,
                dropout=self.hparams.dropout,
                fc_features=tuple(self.hparams.fc_features) if isinstance(self.hparams.fc_features, (list, tuple)) else (384, 384),
            )

    def forward(self, features):
        return self.net(features)

    def _ctc_step(self, batch, stage: str):
        """
        Expected batch keys (typical for CTC):
          features: (B, T, F)
          feature_lengths: (B,)
          targets: (B, S) or 1D concatenated
          target_lengths: (B,)
        Your repo may name these slightly differently; if so, rename here.
        """
        # ---- try common key names ----
        features = batch.get("features", None)
        if features is None:
            # some repos call it "x" or "inputs"
            features = batch.get("x", batch.get("inputs"))

        feat_lens = batch.get("feature_lengths", batch.get("input_lengths", batch.get("x_lengths")))
        targets = batch.get("targets", batch.get("y", batch.get("labels")))
        targ_lens = batch.get("target_lengths", batch.get("y_lengths", batch.get("label_lengths")))

        if any(v is None for v in [features, feat_lens, targets, targ_lens]):
            raise KeyError(
                f"Batch keys not recognized. Got keys={list(batch.keys())}. "
                "Open emg2qwerty/lightning.py and match the baseline batch format."
            )

        logits_TBC = self(features)                 # (T, B, C)
        log_probs = logits_TBC.log_softmax(-1)      # (T, B, C)

        loss = self.ctc_loss(
            log_probs,
            targets,
            feat_lens,
            targ_lens,
        )

        self.log(f"{stage}/loss", loss, prog_bar=True)

        # CER (if compute_cer exists)
        try:
            pred = self.decoder(logits_TBC, feat_lens)
            cer = compute_cer(pred, targets, targ_lens)
            self.log(f"{stage}/CER", cer, prog_bar=True)
        except Exception:
            pass

        return loss

    def training_step(self, batch, batch_idx):
        return self._ctc_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._ctc_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._ctc_step(batch, "test")

    def configure_optimizers(self):
        # Optimizer/scheduler are normally handled in train.py via Hydra in this repo.
        # We return None so Lightning doesn't override the repo's behavior.
        return None
