#!/bin/bash
# Re-run evaluation (plotting only, no training) for the codebook* runs.
# Plots land in <run_dir>/plots_relabelled/ — existing plots are NOT overwritten.
#
# Usage (from the Tokenizer/ directory):
#   bash scripts/replot_codebook_runs.sh

set -e

RUNS_DIR="logs/l1t_tokenization/runs"

for run in codebook512 codebook1024 codebook2048 codebook4096 codebook8192; do
    CKPT="${RUNS_DIR}/${run}/checkpoints/best.ckpt"
    OUT_DIR="$(pwd)/${RUNS_DIR}/${run}/plots_relabelled"
    if [ ! -f "$CKPT" ]; then
        echo "[SKIP] $CKPT not found, skipping $run"
        continue
    fi
    echo "=========================================="
    echo "Evaluating $run"
    echo "  checkpoint: $CKPT"
    echo "  output:     $OUT_DIR"
    echo "=========================================="
    mkdir -p "$OUT_DIR"
    python gabbro/train.py \
        experiment=l1t_tokenization \
        ckpt_path_for_evaluation="${CKPT}" \
        "+callbacks.tokenization_callback.image_path=${OUT_DIR}"
done

echo "Done."
