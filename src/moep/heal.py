"""Recovery tuning (healing) for pruned checkpoints: LoRA distillation from the base model.

Tests whether parameter-efficient recovery suffices in the coding domain, against the
finding of arXiv:2605.28042 that PEFT cannot recover translation specialists. Constraints
verified on this architecture (2026-07-12, tiny-model probe): peft cannot wrap the fused
3D expert parameters (Qwen3_5MoeExperts holds raw nn.Parameter), so LoRA targets every
regular Linear (attention, linear-attention projections, shared expert) and the router
matrices train fully via modules_to_save; routed expert weights stay frozen. That
composition is itself part of the experimental claim and is recorded in the manifest.

The pruned models are small enough (~11B at 25% keep) to train in bf16 without base
quantization, which keeps the sufficiency test free of quantization confounds.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
                "gate_proj", "up_proj", "down_proj", "shared_expert_gate"]
ROUTER_MODULE = "mlp.gate"


def _tokenize_masked(tok, messages: list[dict], max_len: int):
    """Token ids with labels masked to the final assistant response only."""
    full = tok.apply_chat_template(messages, tokenize=True, return_dict=False)
    prompt = tok.apply_chat_template(messages[:-1], tokenize=True, return_dict=False,
                                     add_generation_prompt=True)
    if len(full) > max_len or len(prompt) >= len(full):
        return None
    ids = torch.tensor(full, dtype=torch.long)
    labels = ids.clone()
    labels[: len(prompt)] = -100
    return ids, labels


def heal(model_dir: str, data_path: str, out_dir: str, lr: float = 1e-4, r: int = 64,
         epochs: float = 1.0, seq_len: int = 2048, grad_accum: int = 8,
         warmup: int = 20, seed: int = 17, log=print) -> dict:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer

    from .data import read_jsonl
    from .loader import load_streaming

    torch.manual_seed(seed)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = load_streaming(model_dir, device="cuda:0")
    # fp32 for training, always. In bf16, LoRA merge deltas (~1e-5) and router updates at
    # sane learning rates sit BELOW the ~1e-4 representational step of the weights: the
    # first healing run trained a visibly falling loss and then merge+save rounded every
    # weight back to byte-identical (caught 2026-07-13 by tensor diff). fp32 masters let
    # small consistent updates accumulate; we save in fp32's bf16 rounding only once at
    # the end, when the accumulated delta is large enough to survive it.
    model = model.to(torch.float32)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lcfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05,
                      target_modules=LORA_TARGETS,
                      modules_to_save=[ROUTER_MODULE])
    model = get_peft_model(model, lcfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"heal: trainable {trainable/1e6:.1f}M of {total/1e9:.2f}B "
        f"({100*trainable/total:.2f}%), routed experts frozen by construction")

    rows = read_jsonl(data_path)
    samples = []
    for rw in rows:
        enc = _tokenize_masked(tok, rw["messages"], seq_len)
        if enc is not None:
            samples.append(enc)
    log(f"heal: {len(samples)}/{len(rows)} samples usable at seq_len {seq_len}")

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=lr, weight_decay=0.0)
    n_steps = max(1, int(len(samples) * epochs) // grad_accum)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warmup, 0.5 * (1 + math.cos(math.pi * s / n_steps))))

    model.train()
    order = torch.randperm(len(samples)).tolist()
    losses, step = [], 0
    for epoch_pos in range(int(len(samples) * epochs)):
        ids, labels = samples[order[epoch_pos % len(samples)]]
        out = model(input_ids=ids[None].cuda(), labels=labels[None].cuda())
        (out.loss / grad_accum).backward()
        losses.append(float(out.loss.detach()))
        if (epoch_pos + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0:
                recent = sum(losses[-10 * grad_accum:]) / len(losses[-10 * grad_accum:])
                log(f"heal: step {step}/{n_steps} loss {recent:.4f} "
                    f"lr {sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s")

    log("heal: merging adapters and saving")
    model = model.merge_and_unload()
    model = model.to(torch.bfloat16)  # one rounding, after fp32 accumulation
    out = Path(out_dir)
    # Force sharded+indexed output (max_shard_size) so this always matches the layout
    # every other stage (surgery.py) produces. Without this, HF's save_pretrained can
    # emit a single flat model.safetensors with no index.json for a checkpoint this
    # size, and eval_pair.sh's "is this dir already populated" check (which looks only
    # for model.safetensors.index.json) then concludes the dir is empty and re-runs
    # `moep prune` INTO it from the unhealed base, silently overwriting the healed
    # weights with a fresh unhealed prune (root cause of the 2026-07-13 healing-null:
    # two full training runs were correctly trained then clobbered by this, not by any
    # precision issue - confirmed via scripts/heal_isolate_merge.py).
    model.save_pretrained(out, safe_serialization=True, max_shard_size="4GB")
    assert (out / "model.safetensors.index.json").exists(), \
        "heal: save did not produce an index.json; downstream idempotency checks would silently reprune"
    tok.save_pretrained(out)
    for aux in ("chat_template.jinja", "generation_config.json", "preprocessor_config.json"):
        src = Path(model_dir) / aux
        if src.exists() and not (out / aux).exists():
            (out / aux).write_bytes(src.read_bytes())

    report = {"trainable_params": trainable, "steps": step,
              "final_loss": sum(losses[-50:]) / min(50, len(losses)),
              "first_loss": sum(losses[:50]) / min(50, len(losses)),
              "samples": len(samples), "wall_s": round(time.time() - t0, 1),
              "lora_targets": LORA_TARGETS, "router_fully_trained": True,
              "experts_frozen": True, "out": str(out)}
    (out / "heal_report.json").write_text(json.dumps(report, indent=2))
    log(f"heal: done {report}")
    return report
