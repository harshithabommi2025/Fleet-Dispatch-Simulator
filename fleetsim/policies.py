"""Dispatch policies: decide which idle vehicle serves which waiting request.

Every policy implements `assign(requests, vehicles, now, travel_time)` and returns
a list of (request, vehicle) pairs. Add a new policy by subclassing `DispatchPolicy`
and registering it in POLICIES.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

TravelTimeFn = Callable[[tuple, tuple], float]


class DispatchPolicy:
    name = "base"
    batched = False  # False -> dispatch immediately on every event; True -> only on the batch tick

    def __init__(self, max_pickup_s: float = float("inf"), seed: int = 0):
        self.max_pickup_s = max_pickup_s
        self.rng = np.random.default_rng(seed)

    def assign(self, requests: Sequence, vehicles: Sequence, now: float,
               travel_time: TravelTimeFn) -> list[tuple]:
        raise NotImplementedError


class RandomPolicy(DispatchPolicy):
    """Baseline: oldest request gets a random idle vehicle within range."""
    name = "random"

    def assign(self, requests, vehicles, now, travel_time):
        free = list(vehicles)
        pairs = []
        for req in sorted(requests, key=lambda r: r.t_request):
            in_range = [v for v in free if travel_time(v.pos, req.origin) <= self.max_pickup_s]
            if not in_range:
                continue
            v = in_range[self.rng.integers(len(in_range))]
            free.remove(v)
            pairs.append((req, v))
        return pairs


class GreedyNearestPolicy(DispatchPolicy):
    """First-come-first-served: oldest request gets the closest idle vehicle."""
    name = "greedy"

    def assign(self, requests, vehicles, now, travel_time):
        free = list(vehicles)
        pairs = []
        for req in sorted(requests, key=lambda r: r.t_request):
            if not free:
                break
            best = min(free, key=lambda v: travel_time(v.pos, req.origin))
            if travel_time(best.pos, req.origin) > self.max_pickup_s:
                continue
            free.remove(best)
            pairs.append((req, best))
        return pairs


class BatchedMatchingPolicy(DispatchPolicy):
    """Collect requests for a short window, then solve a min-cost bipartite matching
    (Hungarian algorithm) that minimises total pickup time across the whole batch."""
    name = "batched"
    batched = True

    def __init__(self, max_pickup_s: float = float("inf"), seed: int = 0, wait_weight: float = 0.5):
        super().__init__(max_pickup_s, seed)
        self.wait_weight = wait_weight  # favours requests that have already waited longer

    def assign(self, requests, vehicles, now, travel_time):
        if not requests or not vehicles:
            return []
        big = 1e9
        cost = np.full((len(requests), len(vehicles)), big)
        for i, req in enumerate(requests):
            waited = now - req.t_request
            for j, v in enumerate(vehicles):
                tt = travel_time(v.pos, req.origin)
                if tt <= self.max_pickup_s:
                    cost[i, j] = tt - self.wait_weight * waited
        rows, cols = linear_sum_assignment(cost)
        return [(requests[i], vehicles[j]) for i, j in zip(rows, cols) if cost[i, j] < big / 2]


POLICIES: dict[str, type[DispatchPolicy]] = {
    p.name: p for p in (RandomPolicy, GreedyNearestPolicy, BatchedMatchingPolicy)
}


def get_policy(name: str, **kwargs) -> DispatchPolicy:
    try:
        return POLICIES[name](**kwargs)
    except KeyError:
        raise ValueError(f"Unknown policy '{name}'. Available: {sorted(POLICIES)}") from None
