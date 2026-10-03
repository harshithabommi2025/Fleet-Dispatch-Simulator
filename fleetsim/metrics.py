"""Fleet-efficiency KPIs computed from a finished simulation."""
from __future__ import annotations

import numpy as np

# metric name -> True if higher is better
METRIC_DIRECTION = {
    "service_rate": True,
    "mean_wait_s": False,
    "p90_wait_s": False,
    "utilization": True,
    "deadhead_ratio": False,
    "trips_per_vehicle_hour": True,
}


def summarize(sim) -> dict:
    reqs = sim.requests
    served = [r for r in reqs if r.status == "served"]
    waits = np.array([r.wait_s for r in served]) if served else np.array([np.nan])
    veh = sim.vehicles
    deadhead = sum(v.deadhead_km for v in veh)
    revenue = sum(v.revenue_km for v in veh)
    fleet_hours = len(veh) * sim.cfg.horizon_s / 3600.0
    return {
        "requests": len(reqs),
        "served": len(served),
        "cancelled": sum(r.status == "cancelled" for r in reqs),
        "service_rate": len(served) / len(reqs) if reqs else float("nan"),
        "mean_wait_s": float(np.nanmean(waits)),
        "p50_wait_s": float(np.nanpercentile(waits, 50)),
        "p90_wait_s": float(np.nanpercentile(waits, 90)),
        # share of fleet time spent carrying riders during the service horizon
        "utilization": sum(v.occupied_s for v in veh) / (len(veh) * sim.cfg.horizon_s),
        # share of all driven km that were empty
        "deadhead_ratio": deadhead / (deadhead + revenue) if (deadhead + revenue) else float("nan"),
        "trips_per_vehicle_hour": len(served) / fleet_hours if fleet_hours else float("nan"),
        "deadhead_km": deadhead,
        "revenue_km": revenue,
    }
