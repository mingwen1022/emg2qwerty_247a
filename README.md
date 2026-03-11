# EC ENGR C247A — EMG Keystroke Recognition

**Branch:** `merge-group` (team integration branch)
**Goal:** Replace TDSConv baseline encoder with BiLSTM/BiGRU variants and beat the baseline CER.

**TDSConv baseline:** val CER 18.94, test CER 22.17 (150 epochs) — `logs/2026-02-27/16-29-13/`

**Best result:** BiGRU (dropout=0.5, no SpecAugment) + CTC beam search (beam=50, 6-gram LM) — **val 7.33, test 6.81** — `logs/2026-03-07/01-33-54/` + `logs/2026-03-07/03-17-15/`

---

## Paper Experiment → Log Directory Mapping

All logs under `logs/<date>/<time>/`. Team member runs marked with contributor name.

---

### Table 1: Architecture Comparison (40-epoch screening)

| Model | Val CER | Test CER | Log Directory |
|-------|---------|----------|---------------|
| TDSConv (baseline) | 22.55 | 24.18 | `logs/2026-02-27/16-29-13/` (ep38 ckpt) |
| BiLSTM (h=384, l=2) | 19.87 | 20.08 | `logs/2026-03-01/18-24-33/` |
| BiLSTM (h=384, l=3) | 20.87 | 24.08 | `logs/2026-02-28/17-50-04/` |
| BiLSTM (h=512, l=2) | 19.74 | 19.69 | `logs/2026-02-28/16-46-00/` |
| BiLSTM (h=512, l=3) | 17.88 | 21.55 | `logs/2026-02-28/18-52-31/` |
| BiLSTM + Transformer | 19.03 | 26.56 | `logs/2026-03-01/05-24-51/` |
| Conformer (15.1M) | 30.77 | 26.78 | `logs/2026-03-04/02-33-30/` |
| ConvLSTMConv (k=31, hop=16) | 17.68 | 19.19 | `logs/2026-03-04/04-39-24/` |
| TDS+BiLSTM (40ep) | — | — | `logs/2026-03-10/01-32-38/` |
| BiGRU (screening, hop=48) | 15.06 | 16.56 | `logs/2026-03-05/17-49-16/` |

---

### Table 2: Architecture Comparison (150-epoch full runs)

| Model | Val CER | Test CER | Log Directory |
|-------|---------|----------|---------------|
| TDSConv (baseline) | 18.94 | 22.17 | `logs/2026-02-27/16-29-13/` |
| BiLSTM (h=384, l=2) | 14.55 | 15.76 | `logs/2026-02-28/00-52-40/` |
| BiLSTM (h=512, l=3) | 15.91 | 22.80 | `logs/2026-02-28/23-09-02/` |
| TDS+BiLSTM hybrid (warm init) | 14.58 | 15.47 | `logs/2026-02-28/08-08-49/` → `logs/2026-03-10/08-33-43/` |
| BiLSTM + Transformer | 14.67 | 17.25 | `logs/2026-03-01/08-19-50/` |
| BiLSTM hop=48 | 13.98 | 14.52 | `logs/2026-03-02/22-17-32/` |
| ConvLSTMConv hop=48 | 14.49 | 17.53 | `logs/2026-03-04/06-02-59/` |
| ConvLSTMConv win=16k pad=[900,100] | 13.56 | 14.93 | `logs/2026-03-05/00-06-12/` |
| BiLSTM win=16k pad=[900,100] hop=48 | **13.65** | **14.93** | `logs/2026-03-04/21-03-51/` |
| BiGRU win=16k pad=[900,100] hop=48 | 13.49 | 14.11 | `logs/2026-03-05/19-00-55/` |
| ConvGRUConv win=16k pad=[900,100] | 13.98 | 15.06 | `logs/2026-03-05/20-02-02/` |

**Baseline preprocessing 150ep (hop=16, win=8000, pad=[1800,200]):**

| Model | Val CER | Test CER | Log Directory |
|-------|---------|----------|---------------|
| BiGRU | 16.26 | 53.90 | `logs/2026-03-10/19-06-46/` |
| ConvLSTMConv | 14.60 | 15.99 | `logs/2026-03-10/19-02-13/` |
| ConvGRUConv | 14.84 | 16.92 | `logs/2026-03-10/19-30-59/` |
| TDS+BiLSTM | 17.17 | 18.54 | `logs/2026-03-10/08-33-43/` |

---

### Table 3: Sampling Rate (hop_length) Ablation

Fixed: BiLSTM h=384 l=2, 16ch, 40 epochs.

| hop_length | Rate | Val CER | Test CER | Log Directory |
|-----------|------|---------|----------|---------------|
| 8 | 250 Hz | 26.47 | 25.14 | `logs/2026-03-02/01-13-08/` |
| 16 | 125 Hz (baseline) | 19.87 | 20.08 | `logs/2026-03-02/02-49-12/` |
| 24 | 83 Hz | 18.76 | 18.50 | `logs/2026-03-02/19-24-08/` |
| 32 | 62.5 Hz | 17.50 | 16.99 | `logs/2026-03-02/03-35-51/` |
| 40 | 50 Hz | 18.17 | 18.31 | `logs/2026-03-02/21-25-34/` |
| **48** | **41.7 Hz** | **17.01** | **17.59** | `logs/2026-03-02/19-57-28/` |
| 56 | 35.7 Hz | 17.92 | 18.22 | `logs/2026-03-02/21-45-44/` |
| 64 | 31.25 Hz | 17.68 | 17.53 | `logs/2026-03-02/03-59-50/` |

---

### Table 4: Window Size & Padding Ablation

Fixed: BiLSTM h=384 l=2, hop=48, 40 epochs.

**Window size (pad=[1800,200] fixed):**

| Window | Duration | Val CER | Test CER | Log Directory |
|--------|----------|---------|----------|---------------|
| 4000 | 2s | 17.61 | 16.08 | `logs/2026-03-04/17-37-10/` |
| 8000 (baseline) | 4s | 17.01 | 17.59 | `logs/2026-03-10/00-27-32/` |
| **16000** | **8s** | **15.86** | **16.19** | `logs/2026-03-04/17-58-15/` |

**Padding (window=8000 fixed):**

| Padding | Past ctx | Val CER | Test CER | Log Directory |
|---------|----------|---------|----------|---------------|
| **[900, 100]** | **450ms** | **15.77** | **16.06** | `logs/2026-03-04/18-14-44/` |
| [1800, 200] (baseline) | 900ms | 17.01 | 17.59 | `logs/2026-03-10/00-59-21/` |
| [3600, 400] | 1800ms | 17.39 | 17.22 | `logs/2026-03-04/18-30-30/` |

**Combo experiments:**

| Window | Padding | Val CER | Test CER | Log Directory |
|--------|---------|---------|----------|---------------|
| 4000 | [3600, 400] | 17.10 | 23.17 | `logs/2026-03-04/18-50-48/` |
| **16000** | **[900, 100]** | **15.13** | **16.27** | `logs/2026-03-04/19-17-15/` |
| 16000 | [3600, 400] | 16.08 | 17.25 | `logs/2026-03-04/20-06-29/` |
| 20000 | [900, 100] | 16.98 | 16.00 | `logs/2026-03-07/01-17-34/` |

---

### Table 5: Dropout Ablation

Fixed: BiGRU, win=16k, pad=[900,100], hop=48.

**40-epoch screening:**

| Dropout | Val CER | Test CER | Log Directory |
|---------|---------|----------|---------------|
| 0.0 | 14.93 | 15.32 | `logs/2026-03-06/02-11-05/` |
| 0.1 (baseline) | 15.37 | 15.60 | `logs/2026-03-06/20-24-49/` |
| 0.2 | 14.53 | 15.97 | `logs/2026-03-06/02-25-06/` |
| 0.3 | 14.51 | 15.86 | `logs/2026-03-06/02-39-28/` |
| **0.5** | **14.33** | **14.87** | `logs/2026-03-06/02-53-37/` |
| 0.6 | 14.38 | 14.65 | `logs/2026-03-06/20-44-47/` |
| 0.7 | 14.98 | 15.54 | `logs/2026-03-06/20-58-29/` |

**150-epoch full run:**

| Dropout | Val CER | Test CER | Log Directory |
|---------|---------|----------|---------------|
| 0.1 | 13.49 | 14.11 | `logs/2026-03-05/19-00-55/` |
| **0.5** | **13.03** | **13.59** | `logs/2026-03-06/04-39-16/` |

---

### Table 6: Optimizer Ablation

Fixed: BiGRU best config (win=16k, pad=[900,100], hop=48, dropout=0.5), 40 epochs.

| Config | Val CER | Test CER | Log Directory |
|--------|---------|----------|---------------|
| Adam lr=3e-4 | 19.05 | 20.53 | `logs/2026-03-05/22-54-43/` |
| Adam lr=5e-4 | 15.51 | 17.03 | `logs/2026-03-05/23-08-54/` |
| Adam lr=1e-3 (baseline) | 15.37 | 15.60 | `logs/2026-03-06/02-53-37/` |
| Adam + CosineAnnealing | 17.43 | 29.24 | `logs/2026-03-05/23-23-16/` |
| Adam + WarmupCosine | 15.49 | 15.02 | `logs/2026-03-05/23-37-40/` |
| **AdamW lr=1e-3** | **14.36** | **15.11** | `logs/2026-03-05/23-51-54/` |
| AdamW + CosineAnnealing | 18.25 | 29.63 | `logs/2026-03-06/00-06-08/` |

**150-epoch AdamW:** val 13.36, test 15.32 — `logs/2026-03-06/00-24-50/`

---

### Table 7: SpecAugment / Augmentation Ablation

Fixed: BiGRU best config, dropout=0.5, 40 epochs.

**Coarse augmentation screening:**

| Config | Val CER | Test CER | Log Directory |
|--------|---------|----------|---------------|
| bandrot_wide (offsets ±2) | 16.50 | 16.68 | `logs/2026-03-06/22-21-57/` |
| jitter_high (max_offset=240) | 15.13 | 15.45 | `logs/2026-03-06/22-08-25/` |
| jitter_low (max_offset=60) | 14.53 | 15.52 | `logs/2026-03-06/21-54-54/` |
| specaug_strong (t50_f8) | 15.75 | 16.34 | `logs/2026-03-06/21-41-36/` |
| t25_f4 (baseline) | 14.33 | 14.87 | `logs/2026-03-06/02-53-37/` |
| specaug_weak (t10_f2) | 13.98 | 14.59 | `logs/2026-03-06/21-28-28/` |
| **no_specaug** | **13.36** | **14.20** | `logs/2026-03-07/01-04-26/` |
| + GaussianNoise (std=0.1) | 20.16 | 20.34 | `logs/2026-03-02/17-38-18/` |
| + AmplitudeScale (×0.7~1.3) | 21.02 | 21.98 | `logs/2026-03-02/18-28-20/` |

**Fine-grained SpecAugment screening:**

| Config | time_mask | freq_mask | Val CER | Test CER | Log Directory |
|--------|----------|----------|---------|----------|---------------|
| **no_specaug** | — | — | 13.36 | **14.20** | `logs/2026-03-07/01-04-26/` |
| t5_f2 | 5 | 2 | **13.14** | 14.65 | `logs/2026-03-06/22-43-27/` |
| t10_f1 | 10 | 1 | 13.82 | 15.39 | `logs/2026-03-06/23-23-57/` |
| t10_f2 | 10 | 2 | 13.98 | 14.59 | `logs/2026-03-06/21-28-28/` |
| t10_f3 | 10 | 3 | 13.34 | 14.76 | `logs/2026-03-06/23-37-10/` |
| t15_f2 | 15 | 2 | 14.16 | 15.13 | `logs/2026-03-06/22-56-59/` |
| t20_f2 | 20 | 2 | 14.73 | 14.87 | `logs/2026-03-06/23-10-17/` |
| t25_f4 (baseline) | 25 | 4 | 14.33 | 14.87 | `logs/2026-03-06/02-53-37/` |

**150-epoch no_specaug full run:** val 11.87, test 11.89 — `logs/2026-03-07/01-33-54/`

---

### Table 8: CTC Beam Search Decoder

Applied to BiGRU best checkpoint. 6-gram character LM (wikitext-103).

| Decoder | Val CER | Test CER | Log Directory |
|---------|---------|----------|---------------|
| Greedy (dropout=0.1) | 13.49 | 14.11 | `logs/2026-03-05/19-00-55/` |
| Beam=10 (dropout=0.1) | 9.30 | 8.90 | `logs/2026-03-05/21-43-15/` |
| Beam=25 (dropout=0.1) | 8.75 | 8.88 | `logs/2026-03-05/21-45-53/` |
| **Beam=50 (dropout=0.1)** | **8.46** | **8.69** | `logs/2026-03-05/21-16-02/` |
| Beam=75 (dropout=0.1) | 8.40 | 8.84 | `logs/2026-03-05/21-52-16/` |
| Beam=100 (dropout=0.1) | 8.40 | 8.77 | `logs/2026-03-05/22-13-33/` |
| Beam=50 (dropout=0.5) | 8.82 | 8.17 | `logs/2026-03-06/06-08-22/` |
| **Beam=50 (dropout=0.5, no_specaug)** | **7.33** | **6.81** | `logs/2026-03-07/03-17-15/` |

---

### Table 9: Channel & Data Ablation

**Channel ablation** (BiLSTM h=384, hop=16, 40 epochs):

| Channels/band | in_features | Val CER | Test CER | Log Directory |
|--------------|-------------|---------|----------|---------------|
| 16 (full) | 528 | 19.87 | 20.08 | `logs/2026-03-01/18-24-33/` |
| 12 | 396 | 22.15 | 22.63 | `logs/2026-03-10/18-36-09/` |
| 8 | 264 | 25.88 | 26.35 | `logs/2026-03-01/19-11-21/` |
| 4 | 132 | 36.53 | 37.97 | `logs/2026-03-01/19-55-25/` |
| 2 | 66 | 66.59 | 67.80 | `logs/2026-03-01/20-38-00/` |
| 1 | 33 | 88.04 | 86.90 | `logs/2026-03-01/21-19-36/` |

**Data amount ablation** (BiLSTM h=384, hop=16, 40 epochs):

| Train sessions | Fraction | Val CER | Test CER | Log Directory |
|---------------|----------|---------|----------|---------------|
| 2 | 12.5% | ~100 | ~100 | `logs/2026-03-01/22-16-00/` |
| 4 | 25% | ~100 | ~100 | `logs/2026-03-01/22-30-35/` |
| 8 | 50% | 36.97 | 33.46 | `logs/2026-03-01/22-55-52/` |
| 16 (full) | 100% | 19.87 | 20.08 | `logs/2026-03-01/23-52-19/` |

---

### Team Member Experiments (Transformer baseline comparisons)

| Config | Log Directory | Contributor |
|--------|---------------|-------------|
| Transformer (standard aug) | `logs/2026-03-04/02-35-07/` | team |
| Transformer (no aug) | `logs/2026-03-04/04-40-06/` | team |
| Transformer (heavy aug) | `logs/2026-03-04/06-45-12/` | team |
| Transformer (gaussian aug) | `logs/2026-03-04/08-50-25/` | team |
| Transformer channel=4 | `logs/2026-03-04/10-55-37/` | team |
| Transformer channel=8 | `logs/2026-03-04/12-55-23/` | team |
| Transformer channel=12 | `logs/2026-03-04/14-56-58/` | team |
| Transformer channel=16 | `logs/2026-03-04/17-00-27/` | team |
| Transformer data=25% | `logs/2026-03-04/19-05-47/` | team |
| Transformer data=50% | `logs/2026-03-04/19-41-49/` | team |
| Transformer data=75% | `logs/2026-03-04/20-49-10/` | team |
| Transformer data=100% | `logs/2026-03-04/22-25-42/` | team |
| Transformer resample×8 | `logs/2026-03-05/00-31-04/` | team |
| Transformer resample×4 | `logs/2026-03-05/00-46-41/` | team |
| Transformer resample×2 | `logs/2026-03-05/01-15-04/` | team |
| Transformer resample×1 | `logs/2026-03-05/02-12-28/` | team |

---

## Quick Reference: Best Results

| Stage | Config | Val CER | Test CER |
|-------|--------|---------|----------|
| Baseline | TDSConv, 150ep | 18.94 | 22.17 |
| Architecture | BiLSTM h=384, 150ep | 14.55 | 15.76 |
| + Preprocessing | win=16k, pad=[900,100], hop=48 | 13.65 | 14.93 |
| Switch to GRU | BiGRU same config | 13.49 | 14.11 |
| + Dropout | dropout=0.5 | 13.03 | 13.59 |
| + No SpecAugment | dropout=0.5, no_specaug | 11.87 | 11.89 |
| **+ Beam Search** | **beam=50, 6-gram LM** | **7.33** | **6.81** |

---

## Key Files

| File | Description |
|------|-------------|
| `LSTM_work.md` | Full experiment log with analysis and insights |
| `emg2qwerty/lightning.py` | All module classes (LSTMCTCModule, GRUCTCModule, ConvLSTMConvCTCModule, etc.) |
| `emg2qwerty/modules.py` | LSTMEncoder, BiGRUEncoder, ConvBlock, ChannelSlice |
| `config/model/lstm_ctc.yaml` | BiLSTM config (h=384, l=2) |
| `config/model/gru_ctc.yaml` | BiGRU config (h=384, l=2) |
| `config/model/conv_lstm_conv_ctc.yaml` | Conv+BiLSTM+Conv config |
| `config/model/cgru_ctc.yaml` | CNN+GRU config (Ming's) |
