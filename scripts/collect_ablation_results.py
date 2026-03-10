#!/usr/bin/env python3
"""Collect and summarize ablation study results from log directories.

Scans logs/ for Hydra run directories, extracts val and test metrics from each,
and prints a summary table. Outputs CSV in ablation_sample.csv format.

Usage:
  python scripts/collect_ablation_results.py [logs_dir]
  python scripts/collect_ablation_results.py logs/2025-03-03
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import yaml

WORKER_NAME = "Shu Han"


def find_metrics(run_dir: Path) -> dict | None:
    """Extract val and test metrics from a run.
    run_dir: the Hydra run directory (parent of hydra_configs/).
    Returns dict with 'val' and 'test' keys, each containing metric dicts.
    """
    config_dir = run_dir / "hydra_configs"
    if not config_dir.exists():
        return None

    # Check for results.json (written by train.py)
    results_file = run_dir / "results.json"
    if results_file.exists():
        try:
            with open(results_file) as f:
                data = json.load(f)
            val_metrics = data.get("val_metrics", [])
            test_metrics = data.get("test_metrics", [])
            if val_metrics or test_metrics:
                return {
                    "val": val_metrics[0] if val_metrics else {},
                    "test": test_metrics[0] if test_metrics else {},
                }
        except Exception:
            pass

    # Check for metrics in other JSON files
    for f in run_dir.rglob("*.json"):
        try:
            with open(f) as fp:
                data = json.load(fp)
            if isinstance(data, dict):
                val = {k: v for k, v in data.items() if k.startswith("val/")}
                test = {k: v for k, v in data.items() if k.startswith("test/")}
                if val or test:
                    return {"val": val, "test": test}
        except Exception:
            pass

    # Fallback: check Lightning metrics.csv
    csv_candidates = [run_dir / "metrics.csv"]
    for subdir in ["emg2qwerty", "lightning_logs"]:
        d = run_dir / subdir
        if d.exists():
            csv_candidates.extend(d.rglob("metrics.csv"))
    for csv in csv_candidates:
        try:
            with open(csv) as f:
                lines = f.readlines()
            if not lines:
                continue
            headers = lines[0].strip().split(",")
            last_row = lines[-1].strip().split(",")
            if len(last_row) != len(headers):
                continue
            row = dict(zip(headers, last_row))
            val = {k: float(v) for k, v in row.items() if k.startswith("val/")}
            test = {k: float(v) for k, v in row.items() if k.startswith("test/")}
            if val or test:
                return {"val": val, "test": test}
        except Exception:
            pass

    return None


def get_ablation_label(run_dir: Path) -> tuple[str, str]:
    """Parse overrides and return (ablation_type, short_label).
    ablation_type: preprocessing | channels | data_amount | sampling_rate
    short_label: human-readable variant name
    """
    config_file = run_dir / "hydra_configs" / "overrides.yaml"
    if not config_file.exists():
        return ("other", run_dir.name)

    with open(config_file) as f:
        overrides = yaml.safe_load(f) or []

    ov = {o.split("=", 1)[0]: o.split("=", 1)[1] for o in overrides if "=" in o}

    # Channels ablation
    if "select_channels.num_channels" in ov:
        ch = ov["select_channels.num_channels"]
        return ("channels", f"{ch} channels")

    # Data amount ablation
    if "dataset.train_subset" in ov:
        frac = ov["dataset.train_subset"]
        pct = int(float(frac) * 100) if frac != "1.0" else 100
        return ("data_amount", f"{pct}% data")

    # Sampling rate ablation
    if "resample.factor" in ov:
        factor = int(ov["resample.factor"])
        hz = {1: "2 kHz", 2: "1 kHz", 4: "500 Hz", 8: "250 Hz"}.get(factor, f"{2000//factor} Hz")
        return ("sampling_rate", hz)

    # Preprocessing ablation (transforms=log_spectrogram, preprocessing_*, etc.)
    t = ov.get("transforms", "")
    if "preprocessing_no_aug" in t:
        return ("preprocessing", "no SpecAugment")
    if "preprocessing_heavy_aug" in t:
        return ("preprocessing", "heavy SpecAugment")
    if "preprocessing_gaussian" in t:
        return ("preprocessing", "Gaussian noise")
    if "log_spectrogram" in t:
        return ("preprocessing", "baseline")

    return ("other", str(ov.get("transforms", run_dir.name)))


def _get_total_params_from_checkpoint(run_dir: Path) -> str:
    """Get total param count from checkpoint state_dict. Returns e.g. '32.7M' or ''."""
    try:
        import torch
    except ImportError:
        return ""
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return ""
    ckpt_files = list(ckpt_dir.glob("*.ckpt"))
    if not ckpt_files:
        return ""
    # Prefer last.ckpt, else any checkpoint
    ckpt_path = next((f for f in ckpt_files if f.name == "last.ckpt"), ckpt_files[0])
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        if isinstance(state, dict):
            total = sum(p.numel() for p in state.values() if hasattr(p, "numel"))
            if total > 0:
                return f"{total / 1e6:.1f}M"
    except Exception:
        pass
    return ""


def get_run_config(run_dir: Path) -> dict:
    """Extract config fields for ablation_sample.csv format.
    Returns dict with: architecture, arch_param, total_params, emg_datamodule,
    data_augmentation, channels, train_sessions, effective_rate, optimization,
    others, epochs.
    """
    cfg = run_dir / "hydra_configs" / "config.yaml"
    ov_file = run_dir / "hydra_configs" / "overrides.yaml"
    out = {
        "architecture": "",
        "arch_param": "",
        "total_params": "",
        "emg_datamodule": "",
        "data_augmentation": "",
        "channels": "",
        "train_sessions": "",
        "effective_rate": "",
        "optimization": "Baseline",
        "others": "",
        "epochs": "",
    }
    ov = {}
    config = {}
    if ov_file.exists():
        with open(ov_file) as f:
            overrides = yaml.safe_load(f) or []
        ov = {o.split("=", 1)[0]: o.split("=", 1)[1] for o in overrides if "=" in o}

    if cfg.exists():
        with open(cfg) as f:
            config = yaml.safe_load(f) or {}
        module = config.get("module", {})
        dm = config.get("datamodule", {})
        trainer = config.get("trainer", {})
        # Architecture from module
        target = module.get("_target_", "") or module.get("_target", "")
        if "Transformer" in str(target):
            out["architecture"] = "Transformer"
            d = module.get("d_model", "")
            n = module.get("nhead", "")
            l = module.get("num_layers", "")
            ffn = module.get("dim_feedforward", "")
            out["arch_param"] = f"Transformer(d={d}, nhead={n}, l={l}, ffn={ffn})"
        else:
            out["architecture"] = (
                target.split(".")[-1].replace("Module", "") if target else ""
            )
            out["arch_param"] = str(module)
        # EMG datamodule
        wl = dm.get("window_length", dm.get("window_length", ""))
        pad = dm.get("padding", [])
        out["emg_datamodule"] = f"windowsize={wl}, padding={pad}" if wl else ""
        # Channels: from overrides, select_channels config, or module.electrode_channels
        # (preprocessing runs use log_spectrogram config which has no select_channels)
        ch = (
            ov.get("select_channels.num_channels")
            or config.get("select_channels", {}).get("num_channels")
            or module.get("electrode_channels")
        )
        out["channels"] = str(ch) if ch else ""
        # Epochs
        out["epochs"] = str(ov.get("trainer.max_epochs") or trainer.get("max_epochs", ""))
        # Train sessions (from dataset)
        ds = config.get("dataset", {})
        train_list = ds.get("train", [])
        out["train_sessions"] = str(len(train_list)) if isinstance(train_list, list) else ""

    # Total params: from checkpoint when available
    out["total_params"] = _get_total_params_from_checkpoint(run_dir)

    # Effective rate: spectrogram frame rate = 2000/(factor*16) = 125/factor Hz (matches sample)
    # Preprocessing (log_spectrogram) has no resample → treat as factor=1 → 125Hz
    factor = ov.get("resample.factor")
    if factor is None and cfg.exists():
        resample = config.get("resample", {})
        factor = resample.get("factor") if isinstance(resample, dict) else None
    if factor is None or factor == "":
        factor = 1  # No resample = full rate (preprocessing aug configs)
    if factor:
        try:
            f = int(factor)
            # 125Hz for factor=1, 62.5Hz for factor=2, etc.
            hz = 125 / f
            out["effective_rate"] = f"{hz:.0f}Hz" if hz == int(hz) else f"{hz:.1f}Hz"
        except (ValueError, ZeroDivisionError):
            out["effective_rate"] = str(factor)

    return out


def get_data_augmentation(abl_type: str, label: str) -> str:
    """Data augmentation column: only varies for preprocessing ablation."""
    if abl_type == "preprocessing":
        return label if label else "Baseline"
    return "Baseline"


def parse_ablation_log(ablation_log: Path) -> list[dict]:
    """Parse ablation.log for val and test metrics (when results.json not present).
    Returns list of dicts with 'val' and 'test' keys.
    """
    if not ablation_log.exists():
        return []
    text = ablation_log.read_text()
    pattern = r"['\"]?(val|test)/(\w+)['\"]?\s*:\s*([\d.]+)"
    all_matches = re.findall(pattern, text)
    # Each run outputs 5 val + 5 test = 10 metrics
    results = []
    for i in range(0, len(all_matches), 10):
        chunk = all_matches[i : i + 10]
        if len(chunk) < 10:
            break
        metrics = {"val": {}, "test": {}}
        for phase, name, val in chunk:
            metrics[phase][f"{phase}/{name}"] = float(val)
        results.append(metrics)
    return results


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    logs_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else project_root / "logs"
    ablation_log = project_root / "ablation.log"

    if not logs_dir.exists():
        print(f"Logs dir not found: {logs_dir}")
        return 1

    results = []
    config_paths = sorted(logs_dir.rglob("**/config.yaml"))
    for config_path in config_paths:
        run_dir = config_path.parent.parent
        metrics = find_metrics(run_dir)
        if metrics and (metrics["val"] or metrics["test"]):
            abl_type, label = get_ablation_label(run_dir)
            results.append((abl_type, label, metrics, str(run_dir)))

    # Fallback: parse ablation.log for runs without results.json
    if not results and ablation_log.exists():
        parsed = parse_ablation_log(ablation_log)
        run_dirs = sorted(
            [p.parent.parent for p in config_paths],
            key=lambda d: d.stat().st_mtime,
        )[-len(parsed) :]
        for metrics, run_dir in zip(parsed, run_dirs):
            abl_type, label = get_ablation_label(Path(run_dir))
            results.append((abl_type, label, metrics, str(run_dir)))

    if not results:
        print("No results found. Run ablations first.")
        return 0

    # Group by ablation type, sort by test CER within each group
    type_order = ["preprocessing", "channels", "data_amount", "sampling_rate", "other"]
    by_type: dict[str, list] = {t: [] for t in type_order}
    for abl_type, label, metrics, path in results:
        by_type.setdefault(abl_type, []).append((label, metrics, path))

    def fmt(m: dict, keys: list[str]) -> str:
        parts = []
        for k in keys:
            v = m.get(k)
            parts.append(f"{k}: {v:.2f}" if isinstance(v, (int, float)) else f"{k}: —")
        return " | ".join(parts)

    metric_keys = ["loss", "CER", "IER", "DER", "SER"]
    val_keys = [f"val/{k}" for k in metric_keys]
    test_keys = [f"test/{k}" for k in metric_keys]

    type_titles = {
        "preprocessing": "Preprocessing / Augmentation",
        "channels": "Channels",
        "data_amount": "Data Amount",
        "sampling_rate": "Sampling Rate",
        "other": "Other",
    }

    # Test category for ablation_sample format (maps to terminal section order)
    test_categories = {
        "preprocessing": "Preprocessing",
        "channels": "Channels",
        "data_amount": "Data",
        "sampling_rate": "Sampling Rate",
        "other": "Other",
    }

    print("\n" + "=" * 100)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 100)
    for abl_type in type_order:
        items = by_type.get(abl_type, [])
        if not items:
            continue
        items.sort(key=lambda x: x[1].get("test", {}).get("test/CER", float("inf")))
        print(f"\n  {type_titles[abl_type]}")
        print("  " + "-" * 96)
        for label, metrics, path in items:
            print(f"    {label:<20}  Val: {fmt(metrics.get('val', {}), val_keys)}")
            print(f"    {'':20}  Test: {fmt(metrics.get('test', {}), test_keys)}")
    print("\n" + "=" * 100)

    # Write CSV in ablation_sample.csv format
    csv_path = project_root / "ablation_results.csv"
    csv_headers = [
        "",
        "Worker",
        "Test Category",
        "Architecture",
        "Architecture parameter",
        "Total Params",
        "EMGDatamodule",
        "Data Augmentation",
        "Channels",
        "Train Sessions",
        "Effective Rate",
        "Optimization",
        "Others",
        "Epochs",
        "Val CER",
        "Test CER",
    ]

    def get_metric(m: dict, key: str) -> str:
        v = m.get(key)
        return f"{v:.2f}" if isinstance(v, (int, float)) else ""

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([])  # blank first row like sample
        writer.writerow(csv_headers)
        for abl_type in type_order:
            items = by_type.get(abl_type, [])
            if not items:
                continue
            items.sort(key=lambda x: x[1].get("test", {}).get("test/CER", float("inf")))
            test_cat = test_categories.get(abl_type, abl_type)
            for label, metrics, path in items:
                run_dir = Path(path)
                if not run_dir.is_absolute():
                    run_dir = project_root / run_dir
                run_cfg = get_run_config(run_dir)
                val = metrics.get("val", {})
                test = metrics.get("test", {})
                val_cer = get_metric(val, "val/CER")
                test_cer = get_metric(test, "test/CER")
                row = [
                    "",
                    WORKER_NAME,
                    test_cat,
                    run_cfg["architecture"],
                    run_cfg["arch_param"],
                    run_cfg["total_params"],
                    run_cfg["emg_datamodule"],
                    get_data_augmentation(abl_type, label),
                    run_cfg["channels"],
                    run_cfg["train_sessions"],
                    run_cfg["effective_rate"],
                    run_cfg["optimization"],
                    run_cfg["others"],
                    run_cfg["epochs"],
                    val_cer,
                    test_cer,
                ]
                writer.writerow(row)
        writer.writerow([])  # trailing blank rows like sample
        writer.writerow([])
        writer.writerow([])
        writer.writerow([])

    print(f"\nCSV written to {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
