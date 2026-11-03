#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
: "${DATA_ROOT:?Set DATA_ROOT to the directory containing the image datasets.}"
DESCRIPTION_SET="${DESCRIPTION_SET:-gpt4_tsp_paper}"
read -r -a DATASET_LIST <<< "${DATASETS:-dtd imagenet caltech101 eurosat fgvc food101 oxford_flowers oxford_pets stanford_cars sun397 ucf101}"
read -r -a SEED_LIST <<< "${SEEDS:-1 2 3}"
for dataset in "${DATASET_LIST[@]}"; do
  for seed in "${SEED_LIST[@]}"; do
    log_dir="logs/${DESCRIPTION_SET}/base2new/${dataset}/16shot"
    mkdir -p "$log_dir"
    "${PYTHON:-python}" -u main.py --root_path "$DATA_ROOT" --dataset "$dataset" \
      --shots 16 --seed "$seed" --setting base2new \
      --description_set "$DESCRIPTION_SET" --results_dir "${RESULTS_DIR:-results}" \
      "$@" 2>&1 | tee "${log_dir}/seed${seed}.log"
  done
done
