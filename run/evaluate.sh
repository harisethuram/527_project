#!/bin/bash
#SBATCH --job-name=eval-chexpert
#SBATCH --partition=gpu-l40
#SBATCH --account=ark
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hsethu@uw.edu
#SBATCH --output=/gscratch/ark/hari/527_project/s_out/evaluate_%j.out

# set -euo pipefail

# cd /gscratch/ark/hari/527_project
# source activate chexpert

MODELS="cnn,clip,transformer"
BATCH_SIZE="64"
NUM_WORKERS="8"
MODEL_DIR="models/init"
DATA_DIR="data"
OUTPUT_DIR="evaluation_results"

echo "Running evaluation"
echo "  models:      ${MODELS}"
echo "  batch_size:  ${BATCH_SIZE}"
echo "  num_workers: ${NUM_WORKERS}"
echo "  model_dir:   ${MODEL_DIR}"
echo "  data_dir:    ${DATA_DIR}"
echo "  output_dir:  ${OUTPUT_DIR}"

python evaluate.py \
	--models "${MODELS}" \
	--batch_size "${BATCH_SIZE}" \
	--num_workers "${NUM_WORKERS}" \
	--model_dir "${MODEL_DIR}" \
	--data_dir "${DATA_DIR}" \
	--test_root "chexlocalize/chexlocalize/CheXpert/test" \
	--test_labels "chexlocalize/chexlocalize/CheXpert/test_labels.csv" \
	--output_dir "${OUTPUT_DIR}"

echo "Done. Metrics written to ${OUTPUT_DIR}/metrics_all.csv"
