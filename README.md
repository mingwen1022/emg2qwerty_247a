# EC ENGR C247A — EMG Keystroke Recognition

**Team:** Haneol Choi, Ming Wen, Jake Engelberg, Shu Han Ho

**Best result:** BiGRU (dropout=0.5, no SpecAugment) + beam search (beam=50, 6-gram LM) — **test CER 6.81%**

---

## Paper Experiment → Log Directory Mapping

Logs under `logs/<date>/<time>/`. Ming's cluster runs have a `job0_/` subdirectory.

---

### Table 1: 40-epoch Architecture Comparison (baseline preprocessing)

| Architecture | Params | Val CER | Log Directory |
|---|---|---|---|
| TDS-CNN (k=32) | 5.3M | 22.55 | `logs/2026-02-27/16-29-13/` (ep38) |
| BiRNN (h=512, l=3) | 5.0M | 51.66 | `logs/2026-03-02/02-28-02-40epoch-rnn_3layer_512hid/job0_/` |
| CNN+BiRNN+2FC | 9.5M | 32.63 | `logs/2026-03-02/04-05-05-40epoch-rnn-3layer-512hid-2fcl/job0_/` |
| BiLSTM (h=384, l=2) | 8.2M | 19.87 | `logs/2026-03-01/18-24-33/` |
| BiLSTM (h=512, l=3) | 19.1M | 17.88 | `logs/2026-02-28/18-52-31/` |
| BiGRU (h=384, l=2) | 6.4M | 20.56 | `logs/2026-03-10/00-22-57/` |
| TDS+BiLSTM | 13.0M | 17.17 | `logs/2026-03-10/01-32-38/` |
| Conv+BiLSTM+Conv | 9.4M | 17.68 | `logs/2026-03-04/04-39-24/` |
| Conv+BiGRU+Conv | 7.6M | 27.49 | `logs/2026-03-10/17-19-42/` |
| BiLSTM+Transformer (l=2) | 22.3M | 17.32 | `logs/2026-03-10/03-47-00/` |
| Conformer (l=2, k=31) | 15.1M | 30.77 | `logs/2026-03-04/02-33-30/` |
| Transformer (d=768, l=4) | 32.7M | 19.87 | `logs/2026-03-04/02-35-07/` |

---

### Table 2: 150-epoch Confirmation Runs (baseline preprocessing)

| Architecture | Params | Val CER | Test CER | Log Directory |
|---|---|---|---|---|
| TDS-CNN baseline | 5.3M | 18.94 | 22.17 | `logs/2026-02-27/16-29-13/` |
| BiRNN (h=512, l=5) | 8.1M | 45.70 | 37.00 | `logs/2026-03-01/02-42-22-rnn-5layer-512hd-0.05dp/job0_/` |
| CNN+BiRNN+2FC | 9.5M | 31.52 | 25.20 | `logs/2026-03-02/08-22-06-150epoch-crnn/job0_/` |
| BiLSTM (h=384, l=2) | 8.2M | 14.55 | 15.76 | `logs/2026-02-28/00-52-40/` |
| BiLSTM (h=512, l=3) | 19.1M | 15.91 | 22.80 | `logs/2026-02-28/23-09-02/` |
| BiGRU (h=384, l=2) | 6.4M | 16.26 | 53.90 | `logs/2026-03-10/19-06-46/` |
| TDS+BiLSTM | 13.0M | 14.89 | 15.93 | `logs/2026-03-10/08-33-43/` |
| Conv+BiLSTM+Conv | 9.4M | 14.60 | 15.99 | `logs/2026-03-10/19-02-13/` |
| Conv+BiGRU+Conv | 7.6M | 14.84 | 16.92 | `logs/2026-03-10/19-30-59/` |
| BiLSTM+Transformer (l=2) | 22.3M | 14.67 | 17.25 | `logs/2026-03-01/08-19-50/` |

---

### Table 3: Temporal Resolution Ablation

BiLSTM (h=384, l=2, win=8000), 40-epoch screening.

| Rate | Architecture | Val CER | Test CER | Log Directory |
|---|---|---|---|---|
| 250 Hz (hop=8) | BiLSTM | 26.47 | 25.14 | `logs/2026-03-02/01-13-08/` |
| 125 Hz (hop=16) | BiLSTM | 19.87 | 20.08 | `logs/2026-03-02/02-49-12/` |
| 83 Hz (hop=24) | BiLSTM | 18.76 | 18.50 | `logs/2026-03-02/19-24-08/` |
| 62.5 Hz (hop=32) | BiLSTM | 17.50 | 16.99 | `logs/2026-03-02/03-35-51/` |
| **41.7 Hz (hop=48)** | **BiLSTM** | **17.01** | **17.59** | `logs/2026-03-02/19-57-28/` |
| 41.7 Hz (hop=48) | BiLSTM (150-ep) | 13.98 | 14.52 | `logs/2026-03-02/22-17-32/` |
| 31.25 Hz (hop=64) | BiLSTM | 17.68 | 17.53 | `logs/2026-03-02/03-59-50/` |
| 125 Hz (hop=16) | Transformer | 19.87 | 20.94 | `logs/2026-03-04/02-35-07/` |
| 62.5 Hz (hop=32) | Transformer | 15.86 | 17.48 | `logs/2026-03-05/01-15-04/` |

---

### Table 4: Context Design Ablation

BiLSTM (h=384, l=2, hop=48), 40-epoch screening.

| Window / Padding | Notes | Val CER | Test CER | Log Directory |
|---|---|---|---|---|
| win=4000, pad=[1800,200] | short window | 17.61 | 16.08 | `logs/2026-03-04/17-37-10/` |
| win=8000, pad=[1800,200] | baseline | 17.01 | 17.59 | `logs/2026-03-10/00-27-32/` |
| win=16000, pad=[1800,200] | large window | 15.86 | 16.19 | `logs/2026-03-04/17-58-15/` |
| win=8000, pad=[900,100] | smaller pad | 15.77 | 16.06 | `logs/2026-03-04/18-14-44/` |
| win=8000, pad=[3600,400] | excess pad | 17.39 | 17.22 | `logs/2026-03-04/18-30-30/` |
| **win=16000, pad=[900,100]** | **best combo** | **15.13** | **16.27** | `logs/2026-03-04/19-17-15/` |
| win=16000, pad=[900,100] | 150-ep confirm | 13.65 | 14.93 | `logs/2026-03-04/21-03-51/` |

---

### Table 5: Regularization Ablation

BiGRU (hop=48, win=16000, pad=[900,100]).

| Configuration | Horizon | Val CER | Test CER | Log Directory |
|---|---|---|---|---|
| dropout=0.0, SpecAugment | 40-ep | 14.93 | — | `logs/2026-03-06/02-11-05/` |
| dropout=0.1, SpecAugment | 40-ep | 15.37 | — | `logs/2026-03-06/20-24-49/` |
| dropout=0.2, SpecAugment | 40-ep | 14.53 | — | `logs/2026-03-06/02-25-06/` |
| dropout=0.3, SpecAugment | 40-ep | 14.51 | — | `logs/2026-03-06/02-39-28/` |
| **dropout=0.5, SpecAugment** | **40-ep** | **14.33** | — | `logs/2026-03-06/02-53-37/` |
| dropout=0.6, SpecAugment | 40-ep | 14.38 | — | `logs/2026-03-06/20-44-47/` |
| dropout=0.7, SpecAugment | 40-ep | 14.98 | — | `logs/2026-03-06/20-58-29/` |
| dropout=0.1, SpecAugment | 150-ep | 13.49 | 14.11 | `logs/2026-03-05/19-00-55/` |
| dropout=0.5, SpecAugment | 150-ep | 13.03 | 13.59 | `logs/2026-03-06/04-39-16/` |
| dropout=0.5, no SpecAugment | 40-ep | 13.36 | 14.20 | `logs/2026-03-07/01-04-26/` |
| **dropout=0.5, no SpecAugment** | **150-ep** | **11.87** | **11.89** | `logs/2026-03-07/01-33-54/` |

---

### Table 6: Channel Count Ablation

40-epoch screening, 16 sessions, 125 Hz.

| Ch/band | BiLSTM Val | BiLSTM Test | BiLSTM Log | Transformer Val | Transformer Test | Transformer Log |
|---|---|---|---|---|---|---|
| 16 | 19.87 | 20.08 | `logs/2026-03-01/18-24-33/` | 19.87 | 20.94 | `logs/2026-03-04/17-00-27/` |
| 12 | 22.15 | 22.63 | `logs/2026-03-10/18-36-09/` | 26.14 | 26.94 | `logs/2026-03-04/14-56-58/` |
| 8 | 25.88 | 26.35 | `logs/2026-03-01/19-11-21/` | 29.20 | 30.14 | `logs/2026-03-04/12-55-23/` |
| 4 | 36.53 | 37.97 | `logs/2026-03-01/19-55-25/` | 73.75 | 74.53 | `logs/2026-03-04/10-55-37/` |

---

### Table 7: Training Data Amount Ablation

40-epoch screening, 16 channels, 125 Hz.

| Sessions | BiLSTM Val | BiLSTM Test | BiLSTM Log | Transformer Val | Transformer Test | Transformer Log |
|---|---|---|---|---|---|---|
| 16 | 19.87 | 20.08 | `logs/2026-03-01/23-52-19/` | 19.87 | 20.94 | `logs/2026-03-04/22-25-42/` |
| 8 | 36.97 | 33.46 | `logs/2026-03-01/22-55-52/` | 24.88 | 27.44 | `logs/2026-03-04/19-41-49/` |
| 4 | ~100 | ~100 | `logs/2026-03-01/22-30-35/` | 42.27 | 41.60 | `logs/2026-03-04/19-05-47/` |
| 2 | ~100 | ~100 | `logs/2026-03-01/22-16-00/` | 68.87 | 70.44 | — |

---

### Table 8: CTC Beam Search Decoding (beam=50, 6-gram char LM)

| Model / Configuration | Checkpoint | Val CER | Test CER | Log Directory |
|---|---|---|---|---|
| BiGRU, dropout=0.5, SpecAugment | 150-ep | 8.82 | 8.17 | `logs/2026-03-06/06-08-22/` |
| **BiGRU, dropout=0.5, no SpecAugment** | **150-ep** | **7.33** | **6.81** | `logs/2026-03-07/03-17-15/` |
| Raw-CNN+BiGRU, gru_dropout=0.0 | 100-ep | 6.63 | 7.24 | `logs/2026-03-07/00-41-13/` |
| Raw-CNN+BiGRU, gru_dropout=0.5 | 120-ep | 6.72 | 6.94 | `logs/2026-03-07/02-19-52/` |

---

### Table 9: Raw-CNN+BiGRU Ablation (greedy decoding)

Channels [96,192,256], win=20000, stride=12000, pad=[900,200].

| Configuration | Horizon | Val CER | Test CER | Log Directory |
|---|---|---|---|---|
| gru_dropout=0.0 | 40-ep | 10.64 | 12.60 | `logs/2026-03-06/22-56-37-best_dropout0_40epoch/job0_/` |
| gru_dropout=0.5 | 40-ep | 10.31 | 12.45 | `logs/2026-03-07/01-15-00-best-0.5dropout-40epoch/job0_/` |
| gru_dropout=0.0 | 100-ep | 10.01 | 12.62 | `logs/2026-03-07/00-02-13_best_0dropout_100epoch/job0_/` |
| **gru_dropout=0.5** | **120-ep** | **9.93** | **11.48** | `logs/2026-03-07/01-31-16-best-0.5dropout-120epoch/job0_/` |

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
| `emg2qwerty/lightning.py` | All module classes (LSTMCTCModule, GRUCTCModule, ConvLSTMConvCTCModule, etc.) |
| `emg2qwerty/modules.py` | LSTMEncoder, BiGRUEncoder, ConvBlock, ChannelSlice |
| `config/model/lstm_ctc.yaml` | BiLSTM config (h=384, l=2) |
| `config/model/gru_ctc.yaml` | BiGRU config (h=384, l=2) |
| `config/model/conv_lstm_conv_ctc.yaml` | Conv+BiLSTM+Conv config |
| `config/model/cgru_ctc.yaml` | CNN+GRU config |
| `config/model/raw_cnn_gru_ctc.yaml` | Raw-CNN+BiGRU config |
