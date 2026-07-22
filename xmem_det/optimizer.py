import math

import torch
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer_with_prefix_multipliers(model, cfg, base_lr, group_specs):
    named = list(model.named_parameters())
    seen = set()
    groups = []

    weight_decay = float(cfg.OPTIMIZATION.WEIGHT_DECAY)

    for prefixes, mult in group_specs:
        params = []
        for n, p in named:
            if not p.requires_grad:
                continue
            if not n.startswith(prefixes):
                continue
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            params.append(p)
        if params:
            groups.append(
                {
                    "params": params,
                    "lr": float(base_lr) * float(mult),
                    "weight_decay": weight_decay,
                }
            )

    leftovers = []
    for n, p in named:
        if not p.requires_grad:
            continue
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        leftovers.append(p)

    if leftovers:
        groups.append(
            {
                "params": leftovers,
                "lr": float(base_lr),
                "weight_decay": weight_decay,
            }
        )

    if len(groups) == 0:
        raise ValueError("No trainable parameters matched prefixes; optimizer got an empty parameter list")

    return torch.optim.AdamW(groups, lr=float(base_lr), weight_decay=weight_decay)


def build_warmup_cosine_factor_scheduler(optimizer, steps_per_epoch, epochs, lr_start, lr_max, lr_end, warmup_epochs=1):
    total_steps = int(steps_per_epoch) * int(epochs)
    warmup_steps = int(steps_per_epoch) * int(warmup_epochs)
    warmup_steps = max(1, min(warmup_steps, total_steps))

    start_factor = float(lr_start) / float(lr_max)
    end_factor = float(lr_end) / float(lr_max)

    def factor(step):
        step = int(step)
        if warmup_steps >= total_steps:
            return 1.0
        if step < warmup_steps:
            if warmup_steps == 1:
                return 1.0
            x = step / float(warmup_steps - 1)
            return start_factor + (1.0 - start_factor) * x
        t = step - warmup_steps
        T = total_steps - warmup_steps
        if T <= 1:
            return end_factor
        x = t / float(T - 1)
        c = 0.5 * (1.0 + math.cos(math.pi * x))
        return end_factor + (1.0 - end_factor) * c

    return LambdaLR(optimizer, lr_lambda=[factor for _ in optimizer.param_groups])
