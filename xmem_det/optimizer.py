from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import torch
import math
from torch.optim.lr_scheduler import LambdaLR

from config_utils import cfg_req_nn, cfg_req_float, cfg_req_int, _require_pos_int, _require_prob_01
from typing import Any, Dict, List, Tuple

def set_trainable_prefixes(model, prefixes):
    prefixes = tuple(prefixes)
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith(prefixes)


def build_optimizer_trainable_only(model, cfg, lr):
    params = []
    seen = set()
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        params.append(p)

    return torch.optim.SGD(
        params,
        lr=float(lr),
        momentum=float(cfg.OPTIMIZATION.MOMENTUM),
        weight_decay=float(cfg.OPTIMIZATION.WEIGHT_DECAY),
    )

def build_warmup_cosine_scheduler(optimizer, steps_per_epoch, epochs, lr_start=1e-4, lr_max=1e-3, lr_end=1e-6, warmup_epochs=1):

    total_steps = int(steps_per_epoch) * int(epochs)
    warmup_steps = int(steps_per_epoch) * int(warmup_epochs)
    warmup_steps = max(1, min(warmup_steps, total_steps - 1))

    for pg in optimizer.param_groups:
        pg["lr"] = lr_max

    start_factor = float(lr_start) / float(lr_max)

    warmup = LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    decay_steps = total_steps - warmup_steps

    cosine = CosineAnnealingLR(
        optimizer,
        T_max=decay_steps,
        eta_min=lr_end,
    )

    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )


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

def linear_ramp(epoch_idx: int, start_epoch: int, start: float, end: float, ramp_epochs: int) -> float:
    r = int(ramp_epochs)
    _require_pos_int("RAMP_EPOCHS", r)
    x = float(epoch_idx + 1 - int(start_epoch)) / float(r)
    if x < 0.0:
        x = 0.0
    if x > 1.0:
        x = 1.0
    return float(start) + (float(end) - float(start)) * x



def _prob_sched_epoch(phase_cfg: Any, epoch: int, start_epoch: int) -> Dict[str, float]:
    ps = cfg_req_nn(phase_cfg, "PROB_SCHEDULE")

    t = cfg_req_nn(ps, "TEACHER")
    # c = cfg_req_nn(ps, "CORRUPT")
    # tl = cfg_req_nn(ps, "TEACHER_LAST")
    # cl = cfg_req_nn(ps, "CORRUPT_LAST")

    p_teacher = linear_ramp(
        epoch_idx=epoch,
        start_epoch=start_epoch,
        start=_require_prob_01("TEACHER.START", cfg_req_float(t, "START")),
        end=_require_prob_01("TEACHER.END", cfg_req_float(t, "END")),
        ramp_epochs=cfg_req_int(t, "RAMP_EPOCHS"),
    )
    # p_corrupt = linear_ramp(
    #     epoch_idx=epoch,
    #     start_epoch=start_epoch,
    #     start=_require_prob_01("CORRUPT.START", cfg_req_float(c, "START")),
    #     end=_require_prob_01("CORRUPT.END", cfg_req_float(c, "END")),
    #     ramp_epochs=cfg_req_int(c, "RAMP_EPOCHS"),
    # )
    # p_teacher_last = linear_ramp(
    #     epoch_idx=epoch,
    #     start_epoch=start_epoch,
    #     start=_require_prob_01("TEACHER_LAST.START", cfg_req_float(tl, "START")),
    #     end=_require_prob_01("TEACHER_LAST.END", cfg_req_float(tl, "END")),
    #     ramp_epochs=cfg_req_int(tl, "RAMP_EPOCHS"),
    # )
    # p_corrupt_last = linear_ramp(
    #     epoch_idx=epoch,
    #     start_epoch=start_epoch,
    #     start=_require_prob_01("CORRUPT_LAST.START", cfg_req_float(cl, "START")),
    #     end=_require_prob_01("CORRUPT_LAST.END", cfg_req_float(cl, "END")),
    #     ramp_epochs=cfg_req_int(cl, "RAMP_EPOCHS"),
    # )

    return {
        "p_teacher": float(p_teacher),
        # "p_corrupt": float(p_corrupt),
        # "p_teacher_last": float(p_teacher_last),
        # "p_corrupt_last": float(p_corrupt_last),
    }