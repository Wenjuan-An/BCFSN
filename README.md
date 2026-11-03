# BCFSN: Bidirectional Cross-Modal Feature Selection Network for Few-Shot Image Classification

Wenjuan An, Jijie Wu, Shicheng Yang, Xiaoxu Li, Rui Zhu, Zhanyu Ma.

## Abstract

Recent studies have employed large language models (LLMs) to generate class descriptions with fine-grained visual attributes for local image--text matching in few-shot image classification. However, not all attributes in a class description are relevant to a given input image, and the image regions informative for matching may vary with the candidate class. To address these issues, we propose a Bidirectional Cross-Modal Feature Selection Network (BCFSN) that integrates task-specific class descriptions with bidirectional global-to-local matching. Specifically, we introduce a Cross-Modal Sparse Attention mechanism (CMSA), in which global visual features of the current image guide the selection of relevant text tokens from candidate class descriptions, while global textual features of candidate classes guide the selection of relevant image patches. The selected local features from both directions are weighted and aggregated through sparse attention to compute two global-to-local matching scores, which are combined with the global image--text matching score for classification. To support this mechanism, we introduce a Task-Specific Prompting scheme (TSP) that guides an LLM to generate class descriptions emphasizing task-relevant visual attributes based on the characteristics of different datasets. Experiments on 11 benchmark datasets demonstrate that BCFSN achieves competitive overall performance under both base-to-novel generalization and all-to-all few-shot classification settings. Ablation studies confirm the complementary roles of task-specific descriptions and bidirectional feature selection, while visualizations highlight the selected visual and textual features, enhancing the interpretability of the selection process.

## Requirements

Python 3.10+, CUDA, and a compatible PyTorch/torchvision installation.

```bash
pip install -r requirements.txt
```

The provided `clip/` implementation returns both global and local features. The default backbone is CLIP ViT-B/16.

## Datasets

Download and organize the 11 benchmark datasets following the [CoOp dataset instructions](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md). Dataset images are not included in this repository. Set the path to your own dataset directory:

```bash
export DATA_ROOT="/path/to/datasets"
```

Dataset identifiers: `imagenet`, `caltech101`, `oxford_pets`, `oxford_flowers`, `stanford_cars`, `food101`, `fgvc`, `sun397`, `dtd`, `eurosat`, `ucf101`.

The fixed few-shot support splits are included in `dataset/jsonl_16shot/`. Dataset readers also require the original dataset metadata (e.g., `split_zhou_*.json`); follow the CoOp organization. Pre-generated class descriptions are under `description/`.

## Class Descriptions

| Description set | Prompt strategy |
| --- | --- |
| `gpt4_tsp_paper` (default) | Task-specific prompting used in the paper |
| `gpt4_general` | General visual-description prompting |

The scripts use `gpt4_tsp_paper` by default. To use the task-specific descriptions, set:

```bash
export DESCRIPTION_SET=gpt4_tsp_paper
```


## Evaluation

All scripts run the 11 datasets with seeds `1 2 3` by default. Each experiment uses a single GPU, and the scripts run experiments sequentially. Set `DATA_ROOT` to your dataset directory and `CUDA_VISIBLE_DEVICES` to the GPU you want to use:

```bash
export DATA_ROOT="/path/to/datasets"
export CUDA_VISIBLE_DEVICES=0
```

Replace the example path and GPU index with those available on your machine.

**Base-to-Novel (16-shot):**

```bash
bash scripts/run_base2new.sh
```

**All-to-All (0-, 1-, 4-, and 16-shot):**

```bash
bash scripts/run_zero_shot.sh
bash scripts/run_standard.sh
```

Zero-shot uses frozen CLIP and the equal-weight sum of the global, V→T, and T→V matching scores. Few-shot experiments optimize the three classification losses and use the same score fusion at inference. Only final classification results are saved.

The few-shot settings use rank-2 LoRA in Q/K/V projections of both encoders, batch size 32, learning rate `2e-4`, `300 × shots` optimization steps, and a 30% Top-K retention ratio.

To evaluate a subset of datasets (or selected seeds), set `DATASETS` (or `SEEDS`):

```bash
DATASETS="eurosat dtd fgvc" SEEDS="1 2 3" bash scripts/run_zero_shot.sh
```

## Results

```bash
python summarize.py --root results --outdir "summaries/${DESCRIPTION_SET:-gpt4_tsp_paper}" \
  --description-set "${DESCRIPTION_SET:-gpt4_tsp_paper}" --seeds 1 2 3 --table both --strict
```

This generates `Table1_base_to_novel.csv` and `Table2_all_to_all.csv` under `summaries/<description_set>/`. Use the same description set as in evaluation. For Table 1, Base and Novel accuracies are averaged over the three seeds before calculating each HM; the overall HM is calculated from the 11-dataset average Base and Novel accuracies. Table 2 reports the mean classification accuracy across datasets.

## Acknowledgements

This implementation builds on [CLIP](https://github.com/openai/CLIP), [CLIP-LoRA](https://github.com/MaxZanella/CLIP-LoRA), [2SFS](https://github.com/FarinaMatteo/rethinking_fewshot_vlms), and [CoOp/CoCoOp](https://github.com/KaiyangZhou/CoOp). See [THIRD_PARTY.md](THIRD_PARTY.md) for upstream notices.
