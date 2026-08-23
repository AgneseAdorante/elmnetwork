from __future__ import annotations

import math

_SCHEDULE_KEY = "schedule"


def reg_strength_multiplier(
    progress: float,
    shape: str = "sigmoid",
    warmup_frac: float = 0.5,
    floor: float = 0.0,
) -> float:
    if warmup_frac <= 0.0:
        return 1.0
    t = min(max(progress / warmup_frac, 0.0), 1.0)  # 0..1 across the warmup window
    if shape == "linear":
        ramp = t
    elif shape == "sigmoid":
        # logistic centered at t=0.5.
        k = 10.0
        raw = 1.0 / (1.0 + math.exp(-k * (t - 0.5)))
        lo = 1.0 / (1.0 + math.exp(-k * (0.0 - 0.5)))
        hi = 1.0 / (1.0 + math.exp(-k * (1.0 - 0.5)))
        ramp = (raw - lo) / (hi - lo)
    else:
        raise ValueError(f"Unknown reg-schedule shape {shape!r}")
    return floor + (1.0 - floor) * ramp


def scaled_reg_config(reg_config, progress, schedule_cfg=None):
    """
    Regularization-strength scheduling.
    Ramps the `strength` of selected regularizers from near zero up to their
    configured ceiling over the course of training.
    """
    if reg_config is None:
        return None, 1.0

    shape = schedule_cfg.get("shape", "sigmoid") if schedule_cfg else "sigmoid"
    warmup_frac = schedule_cfg.get("warmup_frac", 0.5) if schedule_cfg else 0.5
    floor = schedule_cfg.get("floor", 0.0) if schedule_cfg else 0.0
    mult = reg_strength_multiplier(progress, shape, warmup_frac, floor)

    scaled = {}
    for reg_type, cfg in reg_config.items():
        cfg = dict(cfg)
        do_ramp = bool(cfg.pop(_SCHEDULE_KEY, False))
        if do_ramp:
            cfg["strength"] = float(cfg["strength"]) * mult
        scaled[reg_type] = cfg
    return scaled, mult
