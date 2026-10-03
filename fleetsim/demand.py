"""Ride-request generation: synthetic demand or real trips loaded from CSV."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .config import SimConfig

Point = tuple[float, float]


@dataclass
class Request:
    id: int
    t_request: float
    origin: Point
    dest: Point
    status: str = "pending"              # pending | assigned | served | cancelled
    assigned_at: Optional[float] = None
    picked_up_at: Optional[float] = None
    dropped_at: Optional[float] = None
    vehicle_id: Optional[int] = None
    meta: dict = field(default_factory=dict)

    @property
    def wait_s(self) -> Optional[float]:
        if self.picked_up_at is None:
            return None
        return self.picked_up_at - self.t_request


def _rate_at(t: float, cfg: SimConfig) -> float:
    """Requests per second at time t: one smooth rush-hour peak in the middle of the horizon."""
    base = cfg.demand_rate_per_hour / 3600.0
    phase = math.sin(math.pi * t / cfg.horizon_s)  # 0 -> 1 -> 0
    # Scale so the *average* rate over the horizon stays close to base.
    shape = 1.0 + (cfg.peak_multiplier - 1.0) * phase
    mean_shape = 1.0 + (cfg.peak_multiplier - 1.0) * (2 / math.pi)
    return base * shape / mean_shape


def _sample_point(rng: np.random.Generator, cfg: SimConfig, centers: np.ndarray) -> Point:
    if len(centers) and rng.random() < cfg.hotspot_share:
        c = centers[rng.integers(len(centers))]
        p = rng.normal(c, cfg.city_km * 0.05)
    else:
        p = rng.uniform(0, cfg.city_km, size=2)
    p = np.clip(p, 0, cfg.city_km)
    return float(p[0]), float(p[1])


def synthetic_demand(cfg: SimConfig, rng: np.random.Generator) -> list[Request]:
    """Non-homogeneous Poisson arrivals (thinning) with spatial hotspots."""
    centers = rng.uniform(0.15 * cfg.city_km, 0.85 * cfg.city_km, size=(cfg.hotspots, 2))
    max_rate = _rate_at(cfg.horizon_s / 2, cfg)
    requests: list[Request] = []
    t = 0.0
    while True:
        t += rng.exponential(1.0 / max_rate)
        if t >= cfg.horizon_s:
            break
        if rng.random() > _rate_at(t, cfg) / max_rate:
            continue  # thinning step
        o = _sample_point(rng, cfg, centers)
        d = _sample_point(rng, cfg, centers)
        requests.append(Request(id=len(requests), t_request=t, origin=o, dest=d))
    return requests


def load_csv_demand(path: str, cfg: SimConfig) -> list[Request]:
    """Load trips from a CSV with columns: t_request_s, ox, oy, dx, dy (km)."""
    df = pd.read_csv(path).sort_values("t_request_s")
    df = df[df["t_request_s"] < cfg.horizon_s]
    return [
        Request(id=i, t_request=float(r.t_request_s), origin=(r.ox, r.oy), dest=(r.dx, r.dy))
        for i, r in enumerate(df.itertuples(index=False))
    ]


def build_demand(cfg: SimConfig, seed: int) -> list[Request]:
    rng = np.random.default_rng(seed)
    if cfg.demand_csv:
        return load_csv_demand(cfg.demand_csv, cfg)
    return synthetic_demand(cfg, rng)
