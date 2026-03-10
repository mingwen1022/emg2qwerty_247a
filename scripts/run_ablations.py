#!/usr/bin/env python3
"""Run ablation studies for emg2qwerty (points 2-5).

Usage:
  # Single experiment (variant=0,1,2,3 for first, second, etc.)
  python scripts/run_ablations.py ablation=preprocessing variant=0

  # Sweep: preprocessing variants (point 2)
  python scripts/run_ablations.py ablation=preprocessing

  # Sweep: channel count (point 3)
  python scripts/run_ablations.py ablation=channels

  # Sweep: data amount (point 4)
  python scripts/run_ablations.py ablation=data_amount

  # Sweep: sampling rate (point 5)
  python scripts/run_ablations.py ablation=sampling_rate

  # All ablations (runs 4 sweeps sequentially)
  python scripts/run_ablations.py ablation=all
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config"

    # Ablation sweep definitions: (ablation_name, list of override strings)
    # All ablations use transformer_ctc with 40 epochs (points 2-5)
    # trainer.accelerator=cpu for CPU-only VMs; override with trainer.accelerator=gpu if you have GPU
    BASE_OVERRIDES = ["model=transformer_ctc", "trainer.max_epochs=40", "trainer.accelerator=cpu"]

    ABLATIONS = {
        "preprocessing": [
            "transforms=log_spectrogram",  # baseline: default aug
            "transforms=preprocessing_no_aug",  # no SpecAugment
            "transforms=preprocessing_heavy_aug",  # heavier SpecAugment
            "transforms=preprocessing_gaussian",  # + Gaussian noise
        ],
        "channels": [
            "transforms=ablation select_channels.num_channels=4 module.electrode_channels=4 module.in_features=132",
            "transforms=ablation select_channels.num_channels=8 module.electrode_channels=8 module.in_features=264",
            "transforms=ablation select_channels.num_channels=12 module.electrode_channels=12 module.in_features=396",
            "transforms=ablation select_channels.num_channels=16 module.electrode_channels=16 module.in_features=528",
        ],
        "data_amount": [
            "transforms=ablation dataset.train_subset=0.25",
            "transforms=ablation dataset.train_subset=0.5",
            "transforms=ablation dataset.train_subset=0.75",
            "transforms=ablation dataset.train_subset=1.0",
        ],
        "sampling_rate": [
            "transforms=ablation resample.factor=8",   # 2000/8 = 250 Hz
            "transforms=ablation resample.factor=4",   # 500 Hz
            "transforms=ablation resample.factor=2",   # 1 kHz
            "transforms=ablation resample.factor=1",   # 2 kHz (full)
        ],
    }

    # Parse args
    args = sys.argv[1:]
    if "ablation=" not in " ".join(args):
        print(__doc__)
        return 1

    ablation_arg = next(a for a in args if a.startswith("ablation="))
    ablation_name = ablation_arg.split("=", 1)[1]
    args = [a for a in args if a != ablation_arg]

    if ablation_name == "all":
        # Run all ablation sweeps
        for name, overrides in ABLATIONS.items():
            print(f"\n{'='*60}\nRunning ablation: {name}\n{'='*60}")
            for override in overrides:
                cmd = [
                    sys.executable, "-m", "emg2qwerty.train",
                    "user=single_user",
                    *BASE_OVERRIDES,
                    *override.split(),
                    *args,
                ]
                print(f"  $ {' '.join(cmd)}")
                ret = subprocess.run(cmd, cwd=project_root)
                if ret.returncode != 0:
                    return ret.returncode
        return 0

    if ablation_name not in ABLATIONS:
        print(f"Unknown ablation: {ablation_name}")
        print(f"Available: {list(ABLATIONS.keys())}, all")
        return 1

    overrides = ABLATIONS[ablation_name]

    # Check for variant= to run single experiment
    variant_arg = next((a for a in args if a.startswith("variant=")), None)
    if variant_arg:
        try:
            idx = int(variant_arg.split("=", 1)[1])
            overrides = [overrides[idx]]
        except (ValueError, IndexError):
            pass
        args = [a for a in args if a != variant_arg]

    # Run each override sequentially (each is a full train+eval)
    for override in overrides:
        cmd = [
            sys.executable, "-m", "emg2qwerty.train",
            "user=single_user",
            *BASE_OVERRIDES,
            *override.split(),
            *args,
        ]
        print(f"\n$ {' '.join(cmd)}\n")
        ret = subprocess.run(cmd, cwd=project_root)
        if ret.returncode != 0:
            return ret.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
