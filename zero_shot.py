"""Frozen-CLIP BCFSN zero-shot evaluation: only the final three-branch accuracy."""
import json
from collections import defaultdict
from pathlib import Path
import time

import clip
import torch
import torch.nn.functional as F

import description
from datasets import build_dataset
from datasets.utils import build_data_loader
from fs.utils.eval_utils import attn_sparse, evaluate
from main import get_arguments, reproducible_setup


# Use the same threshold-based Top-K operator as few-shot evaluation.
def _three_branch_scores(image_global, image_local, text_global, text_local, topk_ratio=0.30):
    """Compute global, T->V and V->T raw matching scores; each [B,C].

    All four feature inputs are individually L2-normalized *before* this call;
    weighted aggregate features are not normalized again. The attention scaling
    follows the supplied BCFSN source, i.e. similarities / (D ** -0.5).
    """
    if image_global.ndim != 2 or image_local.ndim != 3 or text_global.ndim != 2:
        raise ValueError("Expected image_global[B,D], image_local[B,P,D], text_global[C,D].")
    b, dim = image_global.shape
    c = text_global.size(0)
    if image_local.size(0) != b or image_local.size(2) != dim or text_global.size(1) != dim:
        raise ValueError("Mismatched feature dimensions.")
    if len(text_local) != c:
        raise ValueError("One local text feature matrix is required for each candidate class.")

    global_score = image_global @ text_global.t()

    # T->V: every candidate class selects its own patches from each image.
    patch_sim = text_global.unsqueeze(0) @ image_local.transpose(1, 2)  # [B,C,P]
    patch_attn = F.softmax(attn_sparse(patch_sim, topk_ratio) / (dim ** -0.5), dim=-1)
    selected_vision = patch_attn @ image_local  # [B,C,D]
    t2v_score = (selected_vision * text_global.unsqueeze(0)).sum(dim=-1)

    # V->T: each image selects its own tokens for every candidate class.
    # Group equal-length class texts to avoid 1000 sequential kernel launches
    # for ImageNet, without padding that would change per-class Top-K counts.
    by_length = defaultdict(list)
    for class_idx, local in enumerate(text_local):
        if local.ndim != 2 or local.size(1) != dim or local.size(0) < 1:
            raise ValueError(f"Invalid local text tokens for class {class_idx}: {tuple(local.shape)}")
        by_length[local.size(0)].append(class_idx)

    v2t_score = torch.empty_like(global_score)
    for indices in by_length.values():
        local_batch = torch.stack([text_local[j] for j in indices], dim=0)  # [G,L,D]
        token_sim = torch.einsum("bd,gld->bgl", image_global, local_batch)
        token_attn = F.softmax(attn_sparse(token_sim, topk_ratio) / (dim ** -0.5), dim=-1)
        selected_text = torch.einsum("bgl,gld->bgd", token_attn, local_batch)
        v2t_score[:, indices] = (selected_text * image_global[:, None, :]).sum(dim=-1)

    return global_score, t2v_score, v2t_score


@torch.inference_mode()
def evaluate_three_branch_fast(args, model, loader, classnames):
    """Use cached class text and grouped token-length math; return one accuracy (%)."""
    descriptions = description.get_descriptions(args.dataset, args.description_set)
    prompts = [descriptions[name.lower()] for name in classnames]
    global_parts, local_text = [], []
    for start in range(0, len(prompts), args.text_batch_size):
        tokens = clip.tokenize(prompts[start:start + args.text_batch_size], truncate=True).cuda()
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            chunk_local, chunk_global = model.encode_text(tokens)
        global_parts.append(F.normalize(chunk_global, dim=-1))
        local_text.extend(F.normalize(item, dim=-1) for item in chunk_local)
    global_text = torch.cat(global_parts, dim=0)

    correct, total = 0, 0
    for images, target in loader:
        images, target = images.cuda(non_blocking=True), target.cuda(non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            local_image, global_image = model.encode_image(images)
        global_image = F.normalize(global_image, dim=-1)
        local_image = F.normalize(local_image, dim=-1)
        global_score, t2v_score, v2t_score = _three_branch_scores(
            global_image, local_image, global_text, local_text, args.topk_ratio)
        final_score = global_score + t2v_score + v2t_score
        correct += int((final_score.argmax(dim=1) == target).sum().item())
        total += int(target.numel())
        if args.log_every > 0 and total % args.log_every < target.numel():
            print(f"[{args.dataset}] evaluated {total} images", flush=True)
    if total == 0:
        raise RuntimeError(f"Empty test split for dataset={args.dataset}")
    return 100.0 * correct / total, total


@torch.inference_mode()
def main(args):
    if args.setting != "standard" or args.shots != 0:
        raise ValueError("Use --shots 0 --setting standard for zero-shot all-to-all.")
    if not 0 < args.topk_ratio <= 1 or args.text_batch_size < 1:
        raise ValueError("Top-K ratio must be in (0,1] and text batch size >= 1")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for inference")
    if not args.root_path:
        raise ValueError("Specify --root_path or set DATA_ROOT to your local dataset directory")

    start_time = time.time()
    description.configure_descriptions(args.description_set)
    reproducible_setup(args.seed)
    model, preprocess = clip.load(args.backbone)
    model = model.cuda().float().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    # A 1-shot JSONL serves only to initialize the fixed test split in the
    # inherited dataset readers. Zero-shot NEVER uses its support images.
    dataset = build_dataset(args.dataset, args.root_path, 1, "standard", args.seed)
    loader = build_data_loader(
        data_source=dataset.test, batch_size=args.test_batch_size,
        is_train=False, tfm=preprocess, shuffle=False, num_workers=args.workers)

    if args.zero_shot_eval == "reference":
        accuracy = evaluate(args, model, loader, classnames=dataset.test_classnames)
        test_count = len(dataset.test)
    else:
        accuracy, test_count = evaluate_three_branch_fast(
            args, model, loader, dataset.test_classnames)

    result = {
        "output_format_version": 2,
        "acc_test": accuracy,
        "method": "bcfsn_zero_shot", "shots": 0, "seed": args.seed,
        "dataset": args.dataset, "setting": args.setting,
        "backbone": args.backbone.replace("/", "-"),
        "exp_name": args.exp_name, "description_set": args.description_set,
        "topk_ratio": args.topk_ratio, "eval_backend": args.zero_shot_eval,
        "test_batch_size": args.test_batch_size, "text_batch_size": args.text_batch_size,
        "num_test_images": test_count, "num_classes": len(dataset.test_classnames),
        "training_steps": 0, "train_images_used": 0,
        "runtime_seconds": time.time() - start_time,
    }
    outdir = (Path(args.results_dir) / args.description_set / "standard" /
              args.backbone.replace("/", "-") / args.dataset / "shots_0" /
              f"seed_{args.seed}" / "bcfsn_zero_shot")
    outdir.mkdir(parents=True, exist_ok=True)
    destination = outdir / f"{args.exp_name}.json"
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"FINAL ZERO-SHOT | {args.dataset} | seed={args.seed} | Accuracy={accuracy:.4f}%", flush=True)
    print(f"Saved final result: {destination}", flush=True)


if __name__ == "__main__":
    main(get_arguments())
