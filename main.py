import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import datetime
import random
import time

import clip
import numpy as np
import torch

from datasets import build_dataset, build_dataloaders
from fs import run_bcfsn
from fs.utils import dump
from description import DEFAULT_DESCRIPTION_SET, DESCRIPTION_SETS, configure_descriptions


def reproducible_setup(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def get_arguments():
    parser = argparse.ArgumentParser(
        description="BCFSN: Bidirectional Cross-Modal Feature Selection Network"
    )

    # basic configs
    parser.add_argument("--seed", default=1, type=int)

    # dataset configs
    parser.add_argument(
        "--root_path",
        type=str,
        default=os.environ.get("DATA_ROOT", ""),
        help="Root directory where datasets are stored (or $DATA_ROOT)."
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--shots", type=int, required=True)

    parser.add_argument("--description_set", default=DEFAULT_DESCRIPTION_SET, choices=DESCRIPTION_SETS,
                        help="Precomputed class-description set.")

    # dataloader configs
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--test_batch_size", default=32, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--text_batch_size", default=128, type=int,
                        help="Number of class descriptions encoded at once in zero-shot inference.")
    parser.add_argument("--log_every", default=1000, type=int,
                        help="Print zero-shot progress every N images (0 disables).")
    parser.add_argument("--zero_shot_eval", choices=["grouped", "reference"], default="grouped",
                        help="Grouped vectorized zero-shot inference, or original per-class reference.")

    # model configs
    parser.add_argument("--backbone", default="ViT-B/16", type=str)

    # training configs
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--wd", default=1e-2, type=float)
    parser.add_argument("--topk_ratio", default=0.30, type=float,
                        help="Fraction of local tokens and patches retained (default: 0.30).")
    parser.add_argument(
        "--n_iters",
        default=300,
        type=int,
        help="Number of optimization steps per shot (default: 300)."
    )

    # evaluation configs
    parser.add_argument(
        "--setting",
        default="base2new",
        type=str,
        choices=["standard", "base2new"],
        help="Evaluation setting."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run a short debug training loop."
    )
    parser.add_argument("--results_dir", default="results", type=str)
    parser.add_argument("--exp_name", default="bcfsn", type=str)

    # LoRA configs used by BCFSN
    parser.add_argument(
        "--position",
        default="all",
        type=str,
        choices=["bottom", "mid", "up", "half-up", "half-bottom", "all", "top3"],
        help="Transformer blocks where LoRA modules are inserted."
    )
    parser.add_argument(
        "--encoder",
        default="both",
        type=str,
        choices=["text", "vision", "both"],
        help="Encoder branch to tune with LoRA."
    )
    parser.add_argument(
        "--params",
        metavar="N",
        type=str,
        nargs="+",
        default=["q", "k", "v"],
        help="Attention projection matrices where LoRA is inserted."
    )
    parser.add_argument("--r", default=2, type=int, help="LoRA rank.")
    parser.add_argument("--alpha", default=1, type=int, help="LoRA scaling factor.")
    parser.add_argument("--dropout_rate", default=0.25, type=float)
    parser.add_argument(
        "--save_path",
        default=None,
        type=str,
        help="Path to save LoRA weights. If None, weights are not saved."
    )
    parser.add_argument(
        "--filename",
        default="bcfsn_lora_weights",
        type=str,
        help="File name for saved LoRA weights."
    )

    return parser.parse_args()


def main(args):
    if args.shots <= 0:
        raise ValueError("Use zero_shot.py for 0-shot; main.py performs only few-shot training.")
    if not args.root_path:
        raise ValueError("Specify --root_path or set DATA_ROOT to your dataset directory.")
    if not (0 < args.topk_ratio <= 1):
        raise ValueError("--topk_ratio must be in (0, 1].")
    configure_descriptions(args.description_set)
    time_in = time.time()
    reproducible_setup(args.seed)

    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()

    # Following the CLIP-LoRA setting.
    logit_scale = 100.0

    dataset = build_dataset(
        dataset=args.dataset,
        root_path=args.root_path,
        shots=args.shots,
        setting=args.setting,
        seed=args.seed,
    )

    train_loader, test_loader = build_dataloaders(args, dataset, preprocess)

    result = run_bcfsn(
        args,
        clip_model,
        logit_scale,
        dataset,
        train_loader,
        test_loader,
    )

    if args.debug:
        print("DEBUG run: no paper-table result is saved.", flush=True)
        return

    total_seconds = int(time.time() - time_in)
    print("Total time (hh:mm:ss) =", str(datetime.timedelta(seconds=total_seconds)))

    result.update({"runtime": total_seconds, "output_format_version": 2})
    output_args = vars(args).copy()
    output_args["results_dir"] = os.path.join(args.results_dir, args.description_set)
    dump(result, output_args, decimals=8)


if __name__ == "__main__":
    main(get_arguments())
