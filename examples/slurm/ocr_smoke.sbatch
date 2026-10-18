#!/bin/bash
#SBATCH --job-name=fyp-ocr-check
#SBATCH --partition=gpu
#SBATCH --gpus=h100-96:1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
# 48G was normalized to six CPUs here; match the live allocation to avoid srun env conflicts.
# Submit from the staged check directory after verifying this GRES type and creating logs/.
source "$HOME/fyp/envs/ocr-cu128/bin/activate"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
srun python -u ocr_smoke.py --bundle bundle --model "$HOME/fyp/models/qwen3-vl-4b-ebb281e" --output "output-$SLURM_JOB_ID"
