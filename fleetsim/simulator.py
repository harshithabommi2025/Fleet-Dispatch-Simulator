"""Discrete-event fleet simulation built on SimPy."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import simpy

from .config import SimConfig
from .demand import Request, build_demand
from .policies import DispatchPolicy, get_policy

IDLE, TO_PICKUP, WITH_RIDER = "idle", "to_pickup", "with_rider"


@dataclass
class Vehicle:
    id: int
    pos: tuple[float, float]
    state: str = IDLE
    occupied_s: float = 0.0     # time spent carrying a rider
    deadhead_km: float = 0.0    # empty driving to pickups
    revenue_km: float = 0.0     # driving with a rider
    trips: int = 0


def manhattan_km(a, b) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class Simulation:
    def __init__(self, cfg: SimConfig, policy: DispatchPolicy | str = "greedy",
                 seed: int = 0, requests: Optional[list[Request]] = None):
        self.cfg = cfg
        self.seed = seed
        self.policy = (get_policy(policy, max_pickup_s=cfg.max_pickup_s, seed=seed)
                       if isinstance(policy, str) else policy)
        self.env = simpy.Environment()
        self.requests = requests if requests is not None else build_demand(cfg, seed)
        self.requests.sort(key=lambda r: r.t_request)

        rng = np.random.default_rng(seed + 10_000)  # vehicles start in the same spots for every policy
        size = self._city_extent()
        self.vehicles = [Vehicle(i, tuple(rng.uniform(0, size, 2))) for i in range(cfg.num_vehicles)]
        self.pending: list[Request] = []

    # ---- helpers -------------------------------------------------------------
    def _city_extent(self) -> float:
        if not self.requests:
            return self.cfg.city_km
        return max(self.cfg.city_km, max(max(r.origin + r.dest) for r in self.requests))

    def travel_time(self, a, b) -> float:
        return manhattan_km(a, b) / self.cfg.speed_kmh * 3600.0

    def idle_vehicles(self) -> list[Vehicle]:
        return [v for v in self.vehicles if v.state == IDLE]

    # ---- processes -----------------------------------------------------------
    def _arrivals(self):
        for req in self.requests:
            yield self.env.timeout(max(0.0, req.t_request - self.env.now))
            self.pending.append(req)
            if not self.policy.batched:
                self._dispatch()

    def _ticker(self):
        while True:
            yield self.env.timeout(self.cfg.dispatch_interval_s)
            self._expire()
            self._dispatch()

    def _expire(self):
        now = self.env.now
        still = []
        for r in self.pending:
            if now - r.t_request > self.cfg.patience_s:
                r.status = "cancelled"
            else:
                still.append(r)
        self.pending = still

    def _dispatch(self):
        idle = self.idle_vehicles()
        if not self.pending or not idle:
            return
        pairs = self.policy.assign(list(self.pending), idle, self.env.now, self.travel_time)
        for req, veh in pairs:
            self.pending.remove(req)
            req.status, req.assigned_at, req.vehicle_id = "assigned", self.env.now, veh.id
            veh.state = TO_PICKUP
            self.env.process(self._trip(veh, req))

    def _trip(self, v: Vehicle, r: Request):
        # Drive empty to the pickup.
        yield self.env.timeout(self.travel_time(v.pos, r.origin))
        v.deadhead_km += manhattan_km(v.pos, r.origin)
        v.pos, v.state, r.picked_up_at = r.origin, WITH_RIDER, self.env.now
        # Drive the rider to the destination.
        ride = self.travel_time(r.origin, r.dest)
        yield self.env.timeout(ride)
        v.occupied_s += ride
        v.revenue_km += manhattan_km(r.origin, r.dest)
        v.pos, v.state, v.trips = r.dest, IDLE, v.trips + 1
        r.status, r.dropped_at = "served", self.env.now
        if not self.policy.batched:
            self._dispatch()

    # ---- run -----------------------------------------------------------------
    def run(self) -> "Simulation":
        self.env.process(self._arrivals())
        self.env.process(self._ticker())
        self.env.run(until=self.cfg.horizon_s + self.cfg.drain_s)
        for r in self.pending:  # anyone still waiting at the end gave up
            r.status = "cancelled"
        self.pending = []
        return self

    @property
    def duration_s(self) -> float:
        return float(self.env.now)
