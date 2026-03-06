# EMG-to-QWERTY Project Report Draft

*This draft intentionally contains only the `Methods` and `Results` sections.*

## Methods

### Experimental Setup

All experiments were conducted in the personalized single-user setting provided by the course starter repository. We used the shared train/validation/test split from the project configuration and reported character error rate (CER) as the primary metric on both validation and test sessions. Lower CER is better. Unless noted otherwise, models were trained with a CTC objective and decoded with the default greedy CTC decoder; beam-search decoding was applied only in the final decoder study.

The raw signal is surface EMG sampled at 2 kHz from two bands, with 16 electrode channels per band. For the main spectrogram-based model family, the training transform stack was `ToTensor -> RandomBandRotation -> TemporalAlignmentJitter -> LogSpectrogram -> SpecAugment`, while validation and test used `ToTensor -> LogSpectrogram`. After spectrogram generation, all encoder-side comparisons shared the same downstream prediction interface: a normalized spectrogram front end, a rotation-invariant MLP projection, an encoder that produced frame-level emissions, and a linear CTC classifier. This kept the loss and decoder interface fixed so that most comparisons isolated encoder or context-design changes rather than unrelated implementation differences.

Window length, stride, and padding are reported in raw EMG samples. At a 2 kHz sampling rate, `hop=48` corresponds to approximately 41.7 spectrogram frames per second, `window_length=20000` corresponds to 10 s of signal, `stride=12000` corresponds to 6 s, and `padding=[900,200]` corresponds to roughly 450 ms of past context plus 100 ms of future context. The main exception to the spectrogram pipeline was the final raw-CNN+GRU model: it used only `ToTensor`, removed the handcrafted spectrogram and augmentation stack, and learned temporal downsampling directly from raw EMG with 1D convolutions.

### Model Families and Comparison Axes

We compared several encoder families. The reference system was the course baseline TDS CNN. Recurrent alternatives included BiLSTM and BiGRU encoders, as well as hybrid variants such as TDS+BiLSTM, ConvLSTMConv, and ConvGRUConv. We also evaluated attention-heavy alternatives, including a BiLSTM+Transformer stack and a Conformer, to test whether stronger global context modeling would help in this small-data regime. Finally, we introduced a raw-input model that replaced the handcrafted spectrogram transform with a three-layer 1D CNN followed by a two-layer BiGRU.

The experiments were organized around four comparison axes. First, we performed broad architecture selection to identify which encoder family was worth pursuing. Second, we ran recurrent-focused ablations over temporal resolution, context length, hidden size, layer depth, and the addition of convolutional blocks around the recurrent core. Third, we tested data and signal availability through channel-count and training-session ablations. Fourth, we separated encoder improvements from decoder improvements by applying beam-search decoding to the best spectrogram BiGRU checkpoint.

Vanilla BiRNN and early transformer-heavy control runs were kept in the experiment pool as negative controls. They were useful for showing what did not work well in this regime, but they are not central to the main contribution of this report.

### Ablation Protocol

Our ablation strategy used broad 40-epoch screening runs to map the search space, then reran only the most promising candidates for longer training horizons such as 120 or 150 epochs. This was especially important because some larger models looked competitive in early validation but later overfit badly, so the report distinguishes clearly between screening results and confirmation runs. Within each study, we varied one factor family at a time whenever possible and used a fixed backbone as the local reference.

**Table M1. Ablation protocol overview.**

| Study | Variable(s) | Candidate values | Backbone | Evaluation horizon |
|---|---|---|---|---|
| Architecture selection | Encoder family | TDS CNN, BiLSTM, BiGRU, TDS+BiLSTM, BiLSTM+Transformer, Conformer, CNN+RNN controls | Family-dependent | 40-epoch screening, longer reruns for finalists |
| Temporal resolution | Spectrogram hop length | 8, 16, 24, 32, 40, 48, 56, 64 | BiLSTM (h=384, l=2) | 40-epoch sweep, 150-epoch confirmation at `hop=48` |
| Context design | Window length, padding, stride | Window 4k to 20k; padding `[900,100]`, `[1800,200]`, `[3600,400]`; stride 2k to 12k | BiLSTM or BiGRU | 40-epoch sweeps, longer runs for strongest settings |
| Recurrent architecture | Cell type, hidden size, depth, extra conv blocks | LSTM vs GRU; hidden size 384 or 512; 1 to 3 layers; with or without conv blocks | Spectrogram recurrent models | 40-epoch screening, 150-epoch confirmation |
| Data availability | Channels per band, train sessions | Channels 1, 2, 4, 8, 16; sessions 2, 4, 8, 16 | BiLSTM (h=384, l=2) | 40 epochs |
| Decoder | Beam size | Greedy, 10, 25, 50, 75, 100 | Best spectrogram BiGRU checkpoint | Offline decoding |
| Final raw model | Learned downsampling via raw CNN | 3 CNN layers, channels `[64,128,192]`, stride `[3,4,4]`, kernel `[9,9,9]` | Raw-CNN+BiGRU | 40 epochs |

### Decoder and Final Model Design

For decoder-side improvements, we used the repository's existing beam-search decoder with a character-level 6-gram language model (`wikitext-103-6gram-charlm.bin`). The main sweep varied beam size over {10, 25, 50, 75, 100}, with `lm_weight=2.0` and `insertion_bonus=2.0`. This let us measure decoder gain without changing the encoder or retraining the model.

The final raw encoder replaced the handcrafted spectrogram front end with three raw 1D convolution layers, using channels `[64, 128, 192]`, kernels `[9, 9, 9]`, strides `[3, 4, 4]`, paddings `[4, 4, 4]`, and batch normalization. A two-layer bidirectional GRU with hidden size 384 followed the CNN stack. The stride product is `3 x 4 x 4 = 48`, chosen deliberately to match the effective temporal downsampling previously achieved by `hop=48` in the spectrogram pipeline. This model used `window_length=20000`, `stride=12000`, and `padding=[900,200]` raw samples, and should therefore be interpreted as a learned-downsampling counterpart to the best low-frame-rate spectrogram models rather than as an unrelated architecture.

## Results

All CER values below are percentages. We explicitly label 40-epoch runs as screening runs when they are being compared against 120- or 150-epoch confirmation runs.

### General Architecture Selection

The first question was whether the original TDS CNN baseline should remain the main architecture family. Table R1 shows that the answer was no: recurrent encoders consistently outperformed the original CNN baseline, while attention-heavy alternatives such as the Conformer or a Transformer stacked on top of BiLSTM either converged more slowly or generalized worse. This pushed the project toward a recurrent-centered search space.

**Table R1. Representative architecture comparison.**

| Family | Representative configuration | Horizon | Params | Val CER | Test CER | Role |
|---|---|---|---|---|---|---|
| TDSConv baseline | TDS CNN, kernel 32 | 150-epoch full run | 5.3M | 18.94 | 22.17 | Reference CNN baseline |
| BiLSTM | `h=384, l=2`, `hop=16`, `win=8000` | 150-epoch full run | 8.2M | 14.55 | 15.76 | First strong recurrent baseline |
| TDS+BiLSTM | TDS CNN + BiLSTM, warm-started | 150-epoch full run | 13.0M | 14.58 | 15.47 | Hybrid control |
| BiLSTM+Transformer | BiLSTM + 2 Transformer layers | 150-epoch full run | 22.3M | 14.67 | 17.25 | Attention-augmented control |
| Conformer | 2-layer Conformer, kernel 31 | 40-epoch screening | 15.1M | 30.77 | 26.78 | Attention-heavy control |
| BiGRU | `h=384, l=2`, `hop=48`, `win=16000`, `pad=[900,100]` | 150-epoch full run | 6.4M | **13.49** | **14.11** | Best spectrogram encoder |
| CNN+BiRNN control | TDS CNN + BiRNN + 2 FC layers | 150-epoch full run | 9.5M | 31.52 | 25.20 | Negative control |

Two patterns stand out. First, adding recurrence is the decisive architectural change: even the plain BiLSTM already closed most of the gap between the baseline and the later best systems. Second, simply adding more complexity on top of a recurrent model did not help in proportion to parameter count. The TDS+BiLSTM hybrid barely changed the result relative to BiLSTM alone, and the BiLSTM+Transformer stack improved validation only marginally while worsening test CER. As a result, the rest of the report narrows to recurrent architectures and the design choices that matter inside that family.

### Ablation Studies

#### Temporal Resolution and Context Length

The most important low-level design question was how much temporal detail the recurrent encoder actually needed. The baseline inherited a relatively high spectrogram frame rate, but the hop-length sweep in Table R2 shows that lower temporal resolution was consistently better once the model was recurrent.

**Table R2. Hop-length ablation for BiLSTM (`h=384, l=2`), 40-epoch screening. Reference row: `hop=16`.**

| Hop length | Effective rate | Val CER | Test CER | Test delta vs. `hop=16` |
|---|---|---|---|---|
| 8 | 250.0 Hz | 26.47 | 25.14 | +5.06 |
| 16 | 125.0 Hz | 19.87 | 20.08 | 0.00 |
| 24 | 83.3 Hz | 18.76 | 18.50 | -1.58 |
| 32 | 62.5 Hz | 17.50 | 16.99 | -3.09 |
| 40 | 50.0 Hz | 18.17 | 18.31 | -1.77 |
| 48 | 41.7 Hz | **17.01** | 17.59 | -2.49 |
| 56 | 35.7 Hz | 17.92 | 18.22 | -1.86 |
| 64 | 31.3 Hz | 17.68 | 17.53 | -2.55 |

The best screening region was `hop=32` to `hop=64`, with `hop=48` chosen for confirmation because it balanced strong validation performance with a clean 48x downsampling factor. In the longer BiLSTM rerun, `hop=48` improved the full-run result from `14.55 / 15.76` at `hop=16` to `13.98 / 14.52`, confirming that lower frame rate was not just an early-training artifact.

Context design mattered just as much as hop length. Table R3 summarizes the key window, padding, and stride experiments once the search had moved into the low-frame-rate recurrent regime.

**Table R3. Context-design ablations at low frame rate. All rows use `hop=48` when spectrograms are present.**

| Backbone | Window | Padding | Stride | Horizon | Val CER | Test CER |
|---|---|---|---|---|---|---|
| BiLSTM | 8000 | `[1800,200]` | default | 40-epoch screening | 17.01 | 17.59 |
| BiLSTM | 16000 | `[1800,200]` | default | 40-epoch screening | 15.86 | 16.19 |
| BiLSTM | 8000 | `[900,100]` | default | 40-epoch screening | 15.77 | 16.06 |
| BiLSTM | 16000 | `[900,100]` | default | 40-epoch screening | **15.13** | 16.27 |
| BiLSTM | 16000 | `[900,100]` | default | 150-epoch full run | 13.65 | 14.93 |
| BiGRU | 4000 | `[900,100]` | 2000 | 40-epoch screening | 16.64 | 15.34 |
| BiGRU | 12000 | `[900,200]` | 2000 | 40-epoch screening | 14.02 | 49.84 |
| BiGRU | 12000 | `[900,200]` | 6000 | 40-epoch screening | 15.47 | 16.99 |
| BiGRU | 20000 | `[900,200]` | 12000 | 40-epoch screening | 13.83 | **15.09** |

The BiLSTM rows show the cleanest trend: larger windows helped, and smaller padding generally helped as well. The GRU follow-up extends that story. Simply increasing window length was not enough; the amount of overlap also mattered. A `12000` window with a very small stride (`2000`) produced strong validation but catastrophic test CER, whereas the longer `20000` window with a less redundant `12000` stride restored generalization and became the strongest context setting among the GRU screenings. Together, Tables R2 and R3 show that the model benefited from coarser temporal resolution plus longer context, not from denser frame-by-frame detail.

#### Recurrent Architecture Ablations

Once the temporal design had stabilized, the next question was what kind of recurrent encoder should occupy that low-frame-rate setting. Table R4 compares model size, depth, cell type, and the addition of convolutional blocks around the recurrent core.

**Table R4. Recurrent-family ablations.**

| Variant | Context | Horizon | Params | Val CER | Test CER |
|---|---|---|---|---|---|
| BiLSTM `h=384, l=2` | `hop=16`, `win=8000` | 40-epoch screening | 8.2M | 19.87 | 20.08 |
| BiLSTM `h=512, l=2` | `hop=16`, `win=8000` | 40-epoch screening | 12.8M | 19.74 | 19.69 |
| BiLSTM `h=512, l=3` | `hop=16`, `win=8000` | 40-epoch screening | 19.1M | 17.88 | 21.55 |
| BiLSTM `h=384, l=2` | `hop=16`, `win=8000` | 150-epoch full run | 8.2M | 14.55 | 15.76 |
| BiLSTM `h=512, l=3` | `hop=16`, `win=8000` | 150-epoch full run | 19.1M | 15.91 | 22.80 |
| ConvLSTMConv | `hop=48`, `win=16000`, `pad=[900,100]` | 150-epoch full run | 9.4M | 13.56 | 14.93 |
| BiGRU `h=384, l=2` | `hop=48`, `win=16000`, `pad=[900,100]` | 150-epoch full run | 6.4M | **13.49** | **14.11** |
| ConvGRUConv | `hop=48`, `win=16000`, `pad=[900,100]` | 150-epoch full run | 7.6M | 13.98 | 15.06 |

This table shows why the project converged on BiGRU. A larger BiLSTM looked promising during 40-epoch screening, but the longer run made the overfitting problem obvious: the `h=512, l=3` model ended substantially worse on test CER than the smaller BiLSTM. Adding spectrogram-side convolution around the recurrent core also failed to deliver a reliable test benefit. ConvLSTMConv slightly improved validation but only matched or trailed the simpler recurrent baselines on test, and ConvGRUConv was worse than plain BiGRU. The best generalization came from the smaller, simpler BiGRU, which suggests that this single-user regime rewards efficient recurrence more than raw parameter count.

#### Data and Signal Availability Ablations

The final ablation family asked whether the remaining limitation was architectural or informational. We therefore reduced channel count and training-session count while keeping the recurrent backbone fixed.

**Table R5a. Channel-count ablation with BiLSTM (`h=384, l=2`), 40 epochs.**

| Channels per band | Val CER | Test CER | Test delta vs. 16 channels |
|---|---|---|---|
| 16 | **19.87** | **20.08** | 0.00 |
| 8 | 25.88 | 26.35 | +6.27 |
| 4 | 36.53 | 37.97 | +17.89 |
| 2 | 66.59 | 67.80 | +47.72 |
| 1 | 88.04 | 86.90 | +66.82 |

**Table R5b. Training-session ablation with BiLSTM (`h=384, l=2`), 40 epochs.**

| Train sessions | Fraction of full training set | Val CER | Test CER |
|---|---|---|---|
| 16 | 100% | **19.87** | **20.08** |
| 8 | 50% | 36.97 | 33.46 |
| 4 | 25% | ~100 | ~100 |
| 2 | 12.5% | ~100 | ~100 |

Both tables point in the same direction. The model is not operating in an information-rich setting where architecture alone dominates performance. Reducing channels causes a steep and monotonic CER increase, which means the 16-channel signal is genuinely informative rather than heavily redundant. Likewise, reducing the number of training sessions rapidly collapses generalization. This supports the interpretation from the architecture study: the project is fundamentally small-data and signal-limited, so the best-performing models are the ones that make the most efficient use of the available information.

### Decoder Improvement and Final Model Design

The last stage of the project separated decoder gains from encoder gains. We first applied beam search to the best spectrogram BiGRU checkpoint, then compared that decoder-side improvement against the new raw-CNN+GRU encoder. These should not be conflated: the best decoded system and the best greedy encoder are not the same model.

**Table R6a. Beam-search sweep on the best spectrogram BiGRU checkpoint (`val=13.49`, `test=14.11` under greedy decoding).**

| Decoder | Beam size | Val CER | Test CER | Approx. val+test runtime |
|---|---|---|---|---|
| Greedy | 1 | 13.49 | 14.11 | ~1 min |
| Beam search + 6-gram char LM | 10 | 9.30 | 8.90 | ~3 min |
| Beam search + 6-gram char LM | 25 | 8.75 | 8.88 | ~6 min |
| Beam search + 6-gram char LM | 50 | **8.46** | **8.69** | ~15 min |
| Beam search + 6-gram char LM | 75 | 8.40 | 8.84 | ~21 min |
| Beam search + 6-gram char LM | 100 | 8.40 | 8.77 | ~30 min |

Beam search delivered the largest single system-level gain in the entire project. Relative to greedy decoding on the same encoder, `beam=50` reduced validation CER by 5.03 points and test CER by 5.42 points. Larger beams slightly improved validation but provided little or no additional test gain, so `beam=50` was the best speed-accuracy trade-off.

**Table R6b. Final system summary.**

| System role | Configuration | Horizon | Params | Val CER | Test CER | Notes |
|---|---|---|---|---|---|---|
| Best spectrogram encoder under greedy decoding | BiGRU `h=384, l=2`, `hop=48`, `win=16000`, `pad=[900,100]` | 150-epoch full run | 6.4M | 13.49 | 14.11 | Strongest confirmed spectrogram encoder |
| Best greedy encoder overall | Raw 3-layer CNN + 2-layer BiGRU, stride product 48, `win=20000`, `stride=12000`, `pad=[900,200]` | 40-epoch screening | 4.4M | **11.77** | **13.40** | Uses raw EMG only; no handcrafted transform stack |
| Best decoded system | Spectrogram BiGRU + beam search (`beam=50`, 6-gram char LM) | Offline decoding | 6.4M encoder | **8.46** | **8.69** | Decoder gain measured on the spectrogram BiGRU checkpoint |

Table R6b clarifies the final takeaway. The raw-CNN+GRU is the strongest greedy encoder observed so far, and it achieved that result while being smaller than the best spectrogram BiGRU and while removing the handcrafted spectrogram front end entirely. However, the best fully decoded system is still the spectrogram BiGRU plus beam search, because that is the only model for which a decoder sweep was completed. The final conclusion is therefore two-part: learned raw downsampling can beat the spectrogram pipeline at the encoder level, while beam search provides an additional and separately measurable decoder-level gain on top of the best confirmed spectrogram encoder.
