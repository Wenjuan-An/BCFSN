"""BCFSN evaluation: only the final three-branch classification accuracy."""
import importlib
import os
import os.path as osp

import clip
import torch
import torch.nn.functional as F


def attn_sparse(attn_weight, ratio=0.30):
    seq_len = attn_weight.size(-1)
    k = max(int(ratio * seq_len), 1)
    value_inf = torch.tensor(float('-inf')).to(attn_weight.device)
    top_k_values, _ = torch.topk(attn_weight, k=k, dim=-1)
    min_value = top_k_values.min(dim=-1, keepdim=True).values
    sparse_attn_weights = torch.where(attn_weight >= min_value, attn_weight, value_inf)
    return sparse_attn_weights


@torch.no_grad()
def evaluate(args, clip_model, loader, classnames):
    """Return one top-1 accuracy (%) using Global + T->V + V->T scores.

    The class-dependent scores, masked Top-K operator, L2-normalization, and
    scaling are unchanged from the supplied original reference evaluator.
    """
    clip_model.eval()
    descriptions = getattr(importlib.import_module("description"), args.dataset)
    prompts = [descriptions[cl.lower()] for cl in classnames]
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        text_tokens = clip.tokenize(prompts, truncate=True).cuda()
        text_local, text_global = clip_model.encode_text(text_tokens)
    text_global = F.normalize(text_global, dim=-1)
    text_local = [F.normalize(tokens, dim=-1) for tokens in text_local]
    total_correct, total_seen = 0, 0

    for images, target in loader:
        images, target = images.cuda(), target.cuda()
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            image_local, image_global = clip_model.encode_image(images)
        image_global = F.normalize(image_global, dim=-1)
        image_local = F.normalize(image_local, dim=-1)
        dim = image_global.size(1)

        # T->V: each candidate class selects patches in each input image.
        patch_sim = text_global.unsqueeze(0) @ image_local.transpose(1, 2)
        patch_attn = F.softmax(attn_sparse(patch_sim, args.topk_ratio) / (dim ** -0.5), dim=-1)
        selected_visual = patch_attn @ image_local
        t2v_score = (selected_visual.unsqueeze(-2) @ text_global.unsqueeze(0).unsqueeze(-1)).squeeze(-1).squeeze(-1)

        # V->T: each image selects tokens in every candidate class description.
        token_sim = [image_global.unsqueeze(1) @ tokens.unsqueeze(0).transpose(1, 2)
                     for tokens in text_local]
        token_attn = [F.softmax(attn_sparse(sim, args.topk_ratio) / (dim ** -0.5), dim=-1)
                      for sim in token_sim]
        selected_text = torch.cat([weights @ tokens for weights, tokens in zip(token_attn, text_local)], dim=1)
        v2t_score = (selected_text.unsqueeze(-2) @ image_global.unsqueeze(1).unsqueeze(-1)).squeeze(-1).squeeze(-1)

        global_score = image_global @ text_global.t()
        final_score = global_score + t2v_score + v2t_score
        total_correct += int((final_score.argmax(dim=1) == target).sum().item())
        total_seen += int(target.numel())

    if total_seen == 0:
        raise RuntimeError("Test loader contains no images")
    return 100.0 * total_correct / total_seen


def dump(result: dict, args: dict, decimals: int = 8):
    """Write only paper-facing classification metrics and reproducibility metadata."""
    import pandas as pd

    args = args.copy()
    args["backbone"] = args["backbone"].replace("/", "-")
    outpath = osp.join(
        args["results_dir"], args["setting"], args["backbone"],
        args["dataset"], f"shots_{args['shots']}", f"seed_{args['seed']}",
        "bcfsn", args["exp_name"] + ".csv",
    )
    os.makedirs(osp.dirname(outpath), exist_ok=True)
    record = {**result, **args, "method": "bcfsn"}
    df = pd.DataFrame([record])
    for column in df.select_dtypes(include="floating"):
        df[column] = df[column].round(decimals)
    df.to_csv(outpath, index=False)
    print(f"Saved final result: {outpath}", flush=True)
