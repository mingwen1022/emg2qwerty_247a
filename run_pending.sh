#!/bin/bash
# Auto-resume pending experiment pipeline. Skips already-completed experiments.

cd /home/hanchoi/247A
source .venv/bin/activate

LOG=/home/hanchoi/247A/pending_experiment.log

run_if_not_done() {
    local tag=$1
    local outlog=$2
    shift 2
    if grep -q "^${tag} done$" "$LOG" 2>/dev/null; then
        echo "$(date): Skipping ${tag} (already done)" >> "$LOG"
        return
    fi
    echo "$(date): Starting ${tag}" >> "$LOG"
    python -m emg2qwerty.train "$@" > "run_logs/${outlog}" 2>&1
    if [ $? -ne 0 ]; then
        echo "$(date): FAILED ${tag} (exit code $?)" >> "$LOG"
        return 1
    fi
    echo "${tag} done" >> "$LOG"
}

# 40-epoch baseline comparisons
run_if_not_done "gru_125hz_baseline_40ep" gru_125hz_baseline_40ep.log \
    model=gru_ctc trainer.max_epochs=40

run_if_not_done "tds_bilstm_baseline_40ep" tds_bilstm_baseline_40ep.log \
    model=tds_lstm_ctc trainer.max_epochs=40

run_if_not_done "conv_gru_conv_baseline_40ep" conv_gru_conv_baseline_40ep.log \
    model=conv_gru_conv_ctc trainer.max_epochs=40

run_if_not_done "bilstm_transformer_baseline_40ep" bilstm_transformer_baseline_40ep.log \
    model=lstm_transformer_ctc trainer.max_epochs=40

# Full 150-epoch runs
run_if_not_done "tds_bilstm_150ep" tds_bilstm_150ep.log \
    model=tds_lstm_ctc trainer.max_epochs=150

# Channel ablation (12ch, others already done)
run_if_not_done "channel_ablation_12ch" channel_ablation_12ch.log \
    model=lstm_ctc trainer.max_epochs=40 module.in_features=396

# 150-epoch re-runs with baseline preprocessing (hop=16, win=8000, pad=[1800,200])
run_if_not_done "bigru_baseline_150ep" bigru_baseline_150ep.log \
    model=gru_ctc trainer.max_epochs=150

run_if_not_done "conv_lstm_conv_baseline_150ep" conv_lstm_conv_baseline_150ep.log \
    model=conv_lstm_conv_ctc trainer.max_epochs=150

run_if_not_done "conv_gru_conv_baseline_150ep" conv_gru_conv_baseline_150ep.log \
    model=conv_gru_conv_ctc trainer.max_epochs=150 \
    datamodule.window_length=8000 "datamodule.padding=[1800,200]"

echo "$(date): All done!" >> "$LOG"
