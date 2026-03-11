# EC ENGR C247A — EMG Keystroke Recognition

**Branch:** `merge-group` (team integration branch)
**Goal:** Replace TDSConv baseline encoder with BiLSTM/BiGRU variants and beat the baseline CER.

**TDSConv baseline:** val CER 18.94, test CER 22.17 (150 epochs) — `logs/2026-02-27/16-29-13`

**Best result:** BiGRU (dropout=0.5, no SpecAugment) + CTC beam search (beam=50, 6-gram LM) — **val 7.33, test 6.81**

---

## Paper Experiment → Log File Mapping

All logs in `run_logs/` (named files) or `logs/<date>/<time>/` (Hydra timestamped runs).
Team member runs marked with contributor name.

---

### Table 1: Architecture Comparison (40-epoch screening)

| Model | Val CER | Test CER | Log File |
|-------|---------|----------|----------|
| TDSConv (baseline) | 22.55 | 24.18 | `logs/2026-02-27/16-29-13` (40ep checkpoint) |
| BiLSTM (h=384, l=2) | 19.87 | 20.08 | `run_logs/bilstm_baseline_40ep.log` |
| BiLSTM (h=384, l=3) | 20.87 | 24.08 | `run_logs/scale_experiments.log` |
| BiLSTM (h=512, l=2) | 19.74 | 19.69 | `run_logs/scale_experiments.log` |
| BiLSTM (h=512, l=3) | 17.88 | 21.55 | `run_logs/scale_experiments.log` |
| BiLSTM + Transformer | 19.03 | 26.56 | `run_logs/bilstm_transformer_baseline_40ep.log` |
| Conformer (15.1M) | 30.77 | 26.78 | `run_logs/conformer_screening.log` |
| ConvLSTMConv (k=31) | 17.68 | 19.19 | `run_logs/conv_lstm_conv_screening.log` |
| TDS+BiLSTM (warm init) | — | — | (no 40ep run; 150ep only) |
| BiGRU (h=384, l=2) screening | 15.06 | 16.56 | `run_logs/gru_screening.log` |

---

### Table 2: Architecture Comparison (150-epoch full runs)

| Model | Val CER | Test CER | Log File |
|-------|---------|----------|----------|
| TDSConv (baseline) | 18.94 | 22.17 | `logs/2026-02-27/16-29-13` |
| BiLSTM (h=384, l=2) | 14.55 | 15.76 | `logs/2026-02-28/00-52-40` |
| BiLSTM (h=512, l=3) | 15.91 | 22.80 | `run_logs/train_lstm_large.log` |
| TDS+BiLSTM hybrid | 14.58 | 15.47 | `run_logs/tds_bilstm_150ep.log` |
| BiLSTM + Transformer | 14.67 | 17.25 | `run_logs/train_lstm_transformer_full.log` |
| BiLSTM hop=48 | 13.98 | 14.52 | `run_logs/train_hop48_full.log` |
| ConvLSTMConv hop=48 | 14.49 | 17.53 | `run_logs/conv_lstm_conv_hop48.log` |
| ConvLSTMConv win=16k pad=[900,100] | 13.56 | 14.93 | `run_logs/conv_lstm_conv_win16000_pad900_150ep.log` |
| BiLSTM win=16k pad=[900,100] hop=48 | **13.65** | **14.93** | `run_logs/win16000_pad900_150ep.log` |
| BiGRU win=16k pad=[900,100] hop=48 | 13.49 | 14.11 | `run_logs/gru_best.log` |
| ConvGRUConv win=16k pad=[900,100] | 13.98 | 15.06 | `run_logs/conv_gru_conv_best.log` |

**Baseline preprocessing 150ep comparisons** (hop=16, win=8000, pad=[1800,200]):

| Model | Val CER | Test CER | Log File |
|-------|---------|----------|----------|
| BiGRU baseline preprocessing | 16.26 | 53.90 | `run_logs/bigru_baseline_150ep.log` |
| ConvLSTMConv baseline preprocessing | 14.60 | 15.99 | `run_logs/conv_lstm_conv_baseline_150ep.log` |
| ConvGRUConv baseline preprocessing | 14.84 | 16.92 | `run_logs/conv_gru_conv_baseline_150ep.log` |
| TDS+BiLSTM baseline preprocessing | 17.17 | 18.54 | `run_logs/tds_bilstm_150ep.log` |

---

### Table 3: Sampling Rate (hop_length) Ablation

Fixed: BiLSTM h=384 l=2, 16ch, 40 epochs.

| hop_length | Rate | Val CER | Test CER | Log File |
|-----------|------|---------|----------|----------|
| 8 | 250 Hz | 26.47 | 25.14 | `run_logs/sampling_ablation_hop8.log` |
| 16 | 125 Hz (baseline) | 19.87 | 20.08 | `run_logs/sampling_ablation_hop16.log` |
| 24 | 83 Hz | 18.76 | 18.50 | `run_logs/sampling_ablation_hop24.log` |
| 32 | 62.5 Hz | 17.50 | 16.99 | `run_logs/sampling_ablation_hop32.log` |
| 40 | 50 Hz | 18.17 | 18.31 | `run_logs/sampling_ablation_hop40.log` |
| **48** | **41.7 Hz** | **17.01** | **17.59** | `run_logs/sampling_ablation_hop48.log` |
| 56 | 35.7 Hz | 17.92 | 18.22 | `run_logs/sampling_ablation_hop56.log` |
| 64 | 31.25 Hz | 17.68 | 17.53 | `run_logs/sampling_ablation_hop64.log` |

---

### Table 4: Window Size & Padding Ablation

Fixed: BiLSTM h=384 l=2, hop=48, 40 epochs.

**Window size (pad=[1800,200] fixed):**

| Window | Duration | Val CER | Test CER | Log File |
|--------|----------|---------|----------|----------|
| 4000 | 2s | 17.61 | 16.08 | `run_logs/win4000_hop48.log` |
| 8000 (baseline) | 4s | 17.01 | 17.59 | `run_logs/win_experiment.log` |
| **16000** | **8s** | **15.86** | **16.19** | `run_logs/win16000_hop48.log` |

**Padding (window=8000 fixed):**

| Padding | Past ctx | Val CER | Test CER | Log File |
|---------|----------|---------|----------|----------|
| [900, 100] | 450ms | 15.77 | 16.06 | `run_logs/pad900_hop48.log` |
| [1800, 200] (baseline) | 900ms | 17.01 | 17.59 | `run_logs/win_experiment.log` |
| [3600, 400] | 1800ms | 17.39 | 17.22 | `run_logs/pad3600_hop48.log` |

**Combo experiments:**

| Window | Padding | Val CER | Test CER | Log File |
|--------|---------|---------|----------|----------|
| 4000 | [3600, 400] | 17.10 | 23.17 | `run_logs/win4000_pad3600.log` |
| **16000** | **[900, 100]** | **15.13** | **16.27** | `run_logs/win16000_pad900.log` |
| 16000 | [3600, 400] | 16.08 | 17.25 | `run_logs/win16000_pad3600.log` |
| 20000 | [900, 100] | 16.98 | 16.00 | `run_logs/win20000_pad900_40ep.log` |

---

### Table 5: Dropout Ablation

Fixed: BiGRU, win=16k, pad=[900,100], hop=48, 40 epochs.

| Dropout | Val CER | Test CER | Log File |
|---------|---------|----------|----------|
| 0.0 | 14.93 | 15.32 | `run_logs/dropout_0.0.log` |
| 0.1 (baseline) | 15.37 | 15.60 | `run_logs/dropout_0.1.log` |
| 0.2 | 14.53 | 15.97 | `run_logs/dropout_0.2.log` |
| 0.3 | 14.51 | 15.86 | `run_logs/dropout_0.3.log` |
| **0.5** | **14.33** | **14.87** | `run_logs/dropout_0.5.log` |
| 0.6 | 14.38 | 14.65 | `run_logs/dropout_0.6.log` |
| 0.7 | 14.98 | 15.54 | `run_logs/dropout_0.7.log` |

**150-epoch full runs:**

| Dropout | Val CER | Test CER | Log File |
|---------|---------|----------|----------|
| 0.1 | 13.49 | 14.11 | `run_logs/gru_best.log` |
| **0.5** | **13.03** | **13.59** | `run_logs/gru_dropout05_150ep.log` |

---

### Table 6: Optimizer Ablation

Fixed: BiGRU best config (win=16k, pad=[900,100], hop=48, dropout=0.5), 40 epochs.

| Config | Val CER | Test CER | Log File |
|--------|---------|----------|----------|
| Adam lr=3e-4 | 19.05 | 20.53 | `run_logs/opt_adam_lr3e4.log` |
| Adam lr=5e-4 | 15.51 | 17.03 | `run_logs/opt_adam_lr5e4.log` |
| Adam lr=1e-3 (baseline) | 15.37 | 15.60 | `run_logs/dropout_0.5.log` |
| Adam + CosineAnnealing | 17.43 | 29.24 | `run_logs/opt_adam_cosine.log` |
| Adam + WarmupCosine | 15.49 | 15.02 | `run_logs/opt_adam_warmup_cosine.log` |
| **AdamW lr=1e-3** | **14.36** | **15.11** | `run_logs/opt_adamw.log` |
| AdamW + CosineAnnealing | 18.25 | 29.63 | `run_logs/opt_adamw_cosine.log` |

**150-epoch AdamW full run:** val 13.36, test 15.32 → `run_logs/gru_adamw_150ep.log`

---

### Table 7: SpecAugment / Augmentation Ablation

Fixed: BiGRU best config, dropout=0.5, 40 epochs.

**Coarse augmentation screening:**

| Config | Val CER | Test CER | Log File |
|--------|---------|----------|----------|
| bandrot_wide (offsets ±2) | 16.50 | 16.68 | `run_logs/aug_bandrot_wide.log` |
| jitter_high (max_offset=240) | 15.13 | 15.45 | `run_logs/aug_jitter_high.log` |
| jitter_low (max_offset=60) | 14.53 | 15.52 | `run_logs/aug_jitter_low.log` |
| specaug_strong (t50_f8) | 15.75 | 16.34 | `run_logs/aug_specaug_strong.log` |
| t25_f4 (baseline) | 14.33 | 14.87 | `run_logs/aug_specaug_weak.log` (or `dropout_0.5.log`) |
| specaug_weak (t10_f2) | 13.98 | 14.59 | `run_logs/aug_specaug_weak.log` |
| **no_specaug** | **13.36** | **14.20** | `run_logs/aug_no_specaug.log` |
| + GaussianNoise (std=0.1) | 20.16 | 20.34 | `run_logs/augmentation_gaussian.log` |
| + AmplitudeScale (×0.7~1.3) | 21.02 | 21.98 | `run_logs/augmentation_amplitude.log` |

**Fine-grained SpecAugment screening:**

| Config | time_mask | freq_mask | Val CER | Test CER | Log File |
|--------|----------|----------|---------|----------|----------|
| **no_specaug** | — | — | 13.36 | **14.20** | `run_logs/aug_no_specaug.log` |
| t5_f2 | 5 | 2 | **13.14** | 14.65 | `run_logs/aug_specaug_t5_f2.log` |
| t10_f1 | 10 | 1 | 13.82 | 15.39 | `run_logs/aug_specaug_t10_f1.log` |
| t10_f2 | 10 | 2 | 13.98 | 14.59 | `run_logs/aug_specaug_t10_f2.log` |
| t10_f3 | 10 | 3 | 13.34 | 14.76 | `run_logs/aug_specaug_t10_f3.log` |
| t15_f2 | 15 | 2 | 14.16 | 15.13 | `run_logs/aug_specaug_t15_f2.log` |
| t20_f2 | 20 | 2 | 14.73 | 14.87 | `run_logs/aug_specaug_t20_f2.log` |
| t25_f4 (baseline) | 25 | 4 | 14.33 | 14.87 | `run_logs/dropout_0.5.log` |

**150-epoch no_specaug full run:** val 11.87, test 11.89 → `run_logs/gru_no_specaug_150ep.log`

---

### Table 8: CTC Beam Search Decoder

Applied to BiGRU best checkpoint. 6-gram character LM (wikitext-103).

| Decoder | Val CER | Test CER | Log File |
|---------|---------|----------|----------|
| Greedy (dropout=0.1) | 13.49 | 14.11 | `run_logs/gru_best.log` |
| Beam=10 (dropout=0.1) | 9.30 | 8.90 | `run_logs/gru_beam10.log` |
| Beam=25 (dropout=0.1) | 8.75 | 8.88 | `run_logs/gru_beam25.log` |
| **Beam=50 (dropout=0.1)** | **8.46** | **8.69** | `run_logs/gru_beam_search.log` |
| Beam=75 (dropout=0.1) | 8.40 | 8.84 | `run_logs/gru_beam75.log` |
| Beam=100 (dropout=0.1) | 8.40 | 8.77 | `run_logs/gru_beam100.log` |
| Beam=50 (dropout=0.5) | 8.82 | 8.17 | `run_logs/gru_dropout05_beam.log` |
| **Beam=50 (dropout=0.5, no_specaug)** | **7.33** | **6.81** | `run_logs/gru_no_specaug_beam.log` |

---

### Table 9: Channel & Data Ablation

**Channel ablation** (BiLSTM h=384, hop=16, 40 epochs):

| Channels/band | in_features | Val CER | Test CER | Log File |
|--------------|-------------|---------|----------|----------|
| 16 (full) | 528 | 19.87 | 20.08 | `run_logs/channel_ablation_16ch.log` |
| 12 | 396 | 22.15 | 22.63 | `run_logs/channel_ablation_12ch.log` |
| 8 | 264 | 25.88 | 26.35 | `run_logs/channel_ablation_8ch.log` |
| 4 | 132 | 36.53 | 37.97 | `run_logs/channel_ablation_4ch.log` |
| 2 | 66 | 66.59 | 67.80 | `run_logs/channel_ablation_2ch.log` |
| 1 | 33 | 88.04 | 86.90 | `run_logs/channel_ablation_1ch.log` |

**Data amount ablation** (BiLSTM h=384, hop=16, 40 epochs):

| Train sessions | Fraction | Val CER | Test CER | Log File |
|---------------|----------|---------|----------|----------|
| 2 | 12.5% | ~100 | ~100 | `run_logs/data_ablation_2ses.log` |
| 4 | 25% | ~100 | ~100 | `run_logs/data_ablation_4ses.log` |
| 8 | 50% | 36.97 | 33.46 | `run_logs/data_ablation_8ses.log` |
| 16 (full) | 100% | 19.87 | 20.08 | `run_logs/data_ablation_16ses.log` |

---

### Baseline Model Comparisons (Team Contributions)

| Model | Val CER | Test CER | Log File | Contributor |
|-------|---------|----------|----------|-------------|
| GRU (BiGRU, hop=16 baseline) | 20.56 | 73.37 | `run_logs/gru_125hz_baseline_40ep.log` | Han |
| TDS+BiLSTM (40ep) | — | — | `run_logs/tds_bilstm_baseline_40ep.log` | Han |
| ConvGRUConv (40ep) | 27.49 | 27.08 | `run_logs/conv_gru_conv_baseline_40ep.log` | Han |
| BiLSTM+Transformer (40ep) | 17.32 | 36.35 | `run_logs/bilstm_transformer_baseline_40ep.log` | Han |
| GRU (Ming's BiGRU, merge-group) | (see cgru_ctc.yaml) | — | team member log | Ming |
| CNN+GRU (CGRUCTCModule) | (see cgru_ctc.yaml) | — | team member log | Ming |

> **Note:** Ming's GRU and CNN+GRU runs (`config/model/gru_ctc.yaml`, `config/model/cgru_ctc.yaml`) were run by Ming — local logs not available in this repo.

---

## Quick Reference: Best Configs

| Stage | Config | Val CER | Test CER |
|-------|--------|---------|----------|
| Baseline | TDSConv, 150ep | 18.94 | 22.17 |
| Architecture | BiLSTM h=384, 150ep | 14.55 | 15.76 |
| + Preprocessing | win=16k, pad=[900,100], hop=48 | 13.65 | 14.93 |
| Switch to GRU | BiGRU same config | 13.49 | 14.11 |
| + Dropout | dropout=0.5 | 13.03 | 13.59 |
| + No SpecAugment | dropout=0.5, no_specaug | 11.87 | 11.89 |
| + Beam Search | beam=50, 6-gram LM | **7.33** | **6.81** |

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
| `run_logs/` | Named experiment logs |
| `logs/` | Hydra timestamped training logs (with checkpoints) |
