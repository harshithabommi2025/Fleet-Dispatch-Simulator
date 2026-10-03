"""Simulation configuration."""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class SimConfig:
    # Fleet
    num_vehicles: int = 200
    speed_kmh: float = 25.0              # average urban driving speed

    # City (square, coordinates in km)
    city_km: float = 10.0

    # Time
    horizon_s: int = 4 * 3600            # how long riders keep requesting trips
    drain_s: int = 3600                  # extra time to finish in-flight trips

    # Demand (used when demand_csv is not set)
    demand_rate_per_hour: float = 600.0  # average requests per hour
    peak_multiplier: float = 2.0         # rush-hour peak relative to the average
    hotspots: int = 3                    # number of high-demand areas (downtown, airport...)
    hotspot_share: float = 0.6           # fraction of trips starting at a hotspot
    demand_csv: Optional[str] = None     # optional real demand (see scripts/prepare_nyc_data.py)

    # Rider behaviour
    patience_s: int = 600                # rider cancels if not picked up in time

    # Dispatch
    dispatch_interval_s: int = 30        # batching window / retry tick
    max_pickup_s: int = 900              # never send a car further than this

    @classmethod
    def from_dict(cls, d: dict) -> "SimConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SimConfig":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def to_dict(self) -> dict:
        return asdict(self)
