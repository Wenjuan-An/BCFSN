"""BCFSN LoRA adaptation for the Base-to-Novel and All-to-All main tables.

    Training: three CE losses; inference: the direct sum of three matching scores.
    No auxiliary classifier, auxiliary metric, or branch accuracy is reported.
"""
import clip
import torch
import torch.nn.functional as F

from description import get_descriptions
from loralib.utils import apply_lora, get_lora_parameters, mark_only_lora_as_trainable, save_lora
from fs.utils.eval_utils import attn_sparse, evaluate


def run_bcfsn(args, clip_model, logit_scale, dataset, train_loader, test_loader):
    if args.encoder != "both":
        raise ValueError("Main-table BCFSN requires --encoder both.")
    if args.shots <= 0:
        raise ValueError("Few-shot adaptation requires shots > 0; use zero_shot.py for shots=0.")

    descriptions = get_descriptions(args.dataset, args.description_set)
    train_prompts = [descriptions[name.lower()] for name in dataset.val_classnames]
    # Token IDs are constant; text features are re-encoded every step with current LoRA.
    train_tokens = clip.tokenize(train_prompts, truncate=True).cuda()

    lora_modules = apply_lora(args, clip_model)
    clip_model = clip_model.cuda().float()
    mark_only_lora_as_trainable(clip_model)
    total_iters = args.n_iters * args.shots

    optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=args.wd,
                                  betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()
    current_step = 0

    while current_step < total_iters:
        clip_model.train()
        total_loss = 0.0
        seen = 0
        for images, target in train_loader:
            images, target = images.cuda(), target.cuda()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                text_local, text_global = clip_model.encode_text(train_tokens)
                image_local, image_global = clip_model.encode_image(images)
            text_global = F.normalize(text_global, dim=-1)
            text_local = [F.normalize(tokens, dim=-1) for tokens in text_local]
            image_global = F.normalize(image_global, dim=-1)
            image_local = F.normalize(image_local, dim=-1)
            dim = image_global.size(1)

            # T->V: class-global text conditions selection of image patches.
            patch_sim = text_global.unsqueeze(0) @ image_local.transpose(1, 2)
            patch_weights = F.softmax(attn_sparse(patch_sim, args.topk_ratio) / (dim ** -0.5), dim=-1)
            selected_visual = patch_weights @ image_local
            t2v_logits = logit_scale * (selected_visual.unsqueeze(-2) @ text_global.unsqueeze(0).unsqueeze(-1)).squeeze(-1).squeeze(-1)

            # V->T: image-global feature conditions the candidate descriptions.
            token_sim = [image_global.unsqueeze(1) @ tokens.unsqueeze(0).transpose(1, 2)
                         for tokens in text_local]
            token_weights = [F.softmax(attn_sparse(sim, args.topk_ratio) / (dim ** -0.5), dim=-1)
                             for sim in token_sim]
            selected_text = torch.cat([weights @ tokens for weights, tokens in zip(token_weights, text_local)], dim=1)
            v2t_logits = logit_scale * (selected_text.unsqueeze(-2) @ image_global.unsqueeze(1).unsqueeze(-1)).squeeze(-1).squeeze(-1)

            global_logits = logit_scale * (image_global @ text_global.t())
            loss = (F.cross_entropy(t2v_logits, target) +
                    F.cross_entropy(v2t_logits, target) +
                    F.cross_entropy(global_logits, target))

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            seen += int(target.numel())
            total_loss += float(loss.item()) * int(target.numel())
            current_step += 1
            if current_step == total_iters or (args.debug and current_step >= len(train_loader)):
                break
        if seen == 0:
            raise RuntimeError("Training loader contains no images")
        print(f"[{current_step}/{total_iters}] LR={scheduler.get_last_lr()[0]:.6f} "
              f"Loss={total_loss / seen:.4f}", flush=True)
        if args.debug:
            break

    if args.save_path is not None:
        save_lora(args, lora_modules)

    if args.setting == "base2new":
        base_loader, novel_loader = test_loader
        base = evaluate(args, clip_model, base_loader, classnames=dataset.test_classnames)
        novel = evaluate(args, clip_model, novel_loader, classnames=dataset.test_new_classnames)
        hm = 2.0 * base * novel / (base + novel) if base + novel else 0.0
        print(f"FINAL | Base={base:.4f}% | Novel={novel:.4f}% | HM={hm:.4f}%", flush=True)
        return {"acc_test_base": base, "acc_test_new": novel, "hmean": hm}

    accuracy = evaluate(args, clip_model, test_loader, classnames=dataset.test_classnames)
    print(f"FINAL | Accuracy={accuracy:.4f}%", flush=True)
    return {"acc_test": accuracy}
