#!/bin/bash
#SBATCH --job-name=csvlm-baseline
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# EDIT: uncomment and set your cluster's GPU partition (see `sinfo`).
##SBATCH --partition=gpu

# Submit from the project root:   mkdir -p logs && sbatch slurm/baseline.sbatch
set -euo pipefail

# EDIT: activate your environment (conda or venv).
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate csvlm
# source .venv/bin/activate

# Keep the model and dataset cache inside the project folder.
export HF_HOME="$PWD/hf_cache"

# If compute nodes have no internet, first run on the login node:
#   HF_HOME=$PWD/hf_cache python scripts/01_inspect_dataset.py
#   HF_HOME=$PWD/hf_cache huggingface-cli download Qwen/Qwen2-VL-2B-Instruct
# then uncomment the next line.
# export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

nvidia-smi

python scripts/02_run_baseline.py --model Qwen/Qwen2-VL-2B-Instruct --split test --batch-size 8
python scripts/03_evaluate.py outputs/baseline_Qwen2-VL-2B-Instruct_test.jsonl
