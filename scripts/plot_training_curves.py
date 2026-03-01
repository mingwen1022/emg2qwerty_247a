#!/usr/bin/env python3
"""
Plot training curves (loss, CER, etc.) from TensorBoard event files using matplotlib.
Useful for report/paper figures. Reads events from lightning_logs (or any TB logdir).

Usage:
  python scripts/plot_training_curves.py logs/2026-02-25/20-41-01/job0_
  # Figures are saved to <logdir>/figs/ by default. Override with -o.
  python scripts/plot_training_curves.py logs/2026-02-25/20-41-01/job0_ -o /path/to/figs
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def load_scalars(logdir: Path):
    """Load all scalars from TensorBoard event files under logdir. Returns dict tag -> [(step, value), ...]."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        raise SystemExit("Need tensorboard: pip install tensorboard")

    logdir = Path(logdir).resolve()
    if not logdir.is_dir():
        raise FileNotFoundError(f"Not a directory: {logdir}")

    # If logdir is job0_, look for lightning_logs/version_0
    if not (logdir / "version_0").exists() and (logdir / "lightning_logs").exists():
        logdir = logdir / "lightning_logs"
    version_dirs = sorted(logdir.glob("version_*"))
    if not version_dirs:
        event_dirs = [logdir]
    else:
        event_dirs = [version_dirs[0]]  # one run only (e.g. version_0)

    out = {}
    for edir in event_dirs:
        acc = EventAccumulator(str(edir))
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            pts = [(e.step, e.value) for e in acc.Scalars(tag)]
            out[tag] = sorted(pts, key=lambda x: x[0])
    return out


def plot_curves(scalars: dict, out_dir: Path, prefix: str = "", cer_ymax: float | None = None, loss_ymax: float | None = None) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _plot_series(ax, tag: str, steps_vals: list, **kwargs) -> None:
        if not steps_vals:
            return
        steps, vals = zip(*steps_vals)
        kw = dict(alpha=0.9, **kwargs)
        if len(steps) == 1:
            kw.setdefault("marker", "*")
            kw.setdefault("markersize", 12)
            kw.setdefault("linestyle", "None")
        ax.plot(steps, vals, label=tag, **kw)

    # Loss: train, val, test (test is a single point at the end)
    loss_tags = [t for t in scalars if "loss" in t.lower()]
    if loss_tags:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for tag in sorted(loss_tags):
            _plot_series(ax, tag, scalars[tag])
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        if loss_ymax is not None:
            ax.set_ylim(bottom=0, top=loss_ymax)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}loss.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # CER (and optionally other metrics): train, val, test (test = one point)
    cer_tags = [t for t in scalars if "CER" in t or "cer" in t]
    if cer_tags:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for tag in sorted(cer_tags):
            _plot_series(ax, tag, scalars[tag])
        ax.set_xlabel("Step")
        ax.set_ylabel("CER (%)")
        if cer_ymax is not None:
            ax.set_ylim(bottom=0, top=cer_ymax)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}cer.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Optional: one figure with loss + CER (two subplots; includes test as single point)
    if ("train/loss" in scalars or "val/loss" in scalars) and cer_tags:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5), sharex=True)
        for tag in ["train/loss", "val/loss", "test/loss"]:
            if tag in scalars and scalars[tag]:
                _plot_series(ax1, tag, scalars[tag])
        ax1.set_ylabel("Loss")
        if loss_ymax is not None:
            ax1.set_ylim(bottom=0, top=loss_ymax)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        for tag in sorted(cer_tags):
            if scalars[tag]:
                _plot_series(ax2, tag, scalars[tag])
        ax2.set_xlabel("Step")
        ax2.set_ylabel("CER (%)")
        if cer_ymax is not None:
            ax2.set_ylim(bottom=0, top=cer_ymax)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}loss_cer.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Plot training curves from TensorBoard logs")
    ap.add_argument("logdir", type=Path, help="Path to lightning_logs or job0_ (contains lightning_logs)")
    ap.add_argument(
        "--out-dir",
        "-o",
        type=Path,
        default=None,
        help="Output directory for figures (default: <logdir>/figs)",
    )
    ap.add_argument("--prefix", "-p", default="", help="Filename prefix for saved figures")
    ap.add_argument(
        "--cer-ymax",
        type=float,
        default=100.0,
        help="Y-axis max for CER plot (default: 100). Use to focus on convergence when initial CER is very high.",
    )
    ap.add_argument(
        "--loss-ymax",
        type=float,
        default=20.0,
        help="Y-axis max for Loss plot (default: 20). Use to focus on convergence when initial loss is very high.",
    )
    args = ap.parse_args()

    logdir = Path(args.logdir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir is not None else logdir / "figs"

    scalars = load_scalars(logdir)
    if not scalars:
        print("No scalar tags found in", logdir)
        return
    print("Tags found:", list(scalars.keys()))
    plot_curves(scalars, out_dir, prefix=args.prefix, cer_ymax=args.cer_ymax, loss_ymax=args.loss_ymax)
    # Write final test metrics for report
    test_metrics = {k: v[-1][1] for k, v in scalars.items() if k.startswith("test/") and v}
    if test_metrics:
        summary_path = out_dir / f"{args.prefix}test_metrics.txt"
        with open(summary_path, "w") as f:
            f.write("Final test metrics (single run)\n")
            for k, v in sorted(test_metrics.items()):
                f.write(f"  {k}: {v}\n")
        print("Test metrics summary:", summary_path)
    print("Saved figures to", out_dir)


if __name__ == "__main__":
    main()
