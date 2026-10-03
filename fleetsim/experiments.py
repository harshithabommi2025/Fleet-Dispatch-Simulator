"""Multi-seed experiments, confidence intervals, and baseline comparison.

Every policy is run on the *same* seeds, so each seed gives every policy identical
demand and starting vehicle positions (common random numbers). That lets us use a
paired t-test, which detects real differences with far fewer runs.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from .config import SimConfig
from .demand import build_demand
from .metrics import METRIC_DIRECTION, summarize
from .simulator import Simulation


def run_once(cfg: SimConfig, policy: str, seed: int) -> dict:
    sim = Simulation(cfg, policy=policy, seed=seed, requests=build_demand(cfg, seed)).run()
    return {"policy": policy, "seed": seed, **summarize(sim)}


def run_experiment(cfg: SimConfig, policies: Iterable[str], seeds: Iterable[int]) -> pd.DataFrame:
    rows = [run_once(cfg, p, s) for s in seeds for p in policies]
    return pd.DataFrame(rows)


def confidence_intervals(df: pd.DataFrame, metrics=None, level: float = 0.95) -> pd.DataFrame:
    metrics = metrics or list(METRIC_DIRECTION)
    out = []
    for policy, g in df.groupby("policy", sort=False):
        for m in metrics:
            x = g[m].dropna().to_numpy()
            mean = x.mean()
            if len(x) > 1:
                half = stats.t.ppf((1 + level) / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
            else:
                half = float("nan")
            out.append({"policy": policy, "metric": m, "mean": mean,
                        "ci_low": mean - half, "ci_high": mean + half, "n": len(x)})
    return pd.DataFrame(out)


def compare_to_baseline(df: pd.DataFrame, baseline: str, metrics=None,
                        alpha: float = 0.05, regression_pct: float = 2.0) -> pd.DataFrame:
    """For each policy and metric: % change vs baseline, paired t-test p-value,
    and a REGRESSION flag when the policy is significantly worse by > regression_pct %."""
    metrics = metrics or list(METRIC_DIRECTION)
    base = df[df.policy == baseline].set_index("seed")
    out = []
    for policy, g in df[df.policy != baseline].groupby("policy", sort=False):
        g = g.set_index("seed")
        seeds = base.index.intersection(g.index)
        for m in metrics:
            b, x = base.loc[seeds, m].to_numpy(), g.loc[seeds, m].to_numpy()
            pct = (x.mean() - b.mean()) / abs(b.mean()) * 100 if b.mean() else float("nan")
            p = stats.ttest_rel(x, b).pvalue if len(seeds) > 1 and np.any(x != b) else float("nan")
            better = (pct > 0) == METRIC_DIRECTION[m]
            significant = bool(p < alpha) if not np.isnan(p) else False
            if significant and better:
                verdict = "IMPROVED"
            elif significant and abs(pct) > regression_pct:
                verdict = "REGRESSION"
            else:
                verdict = "no significant change"
            out.append({"policy": policy, "metric": m, "baseline_mean": b.mean(),
                        "policy_mean": x.mean(), "pct_change": pct, "p_value": p,
                        "verdict": verdict})
    return pd.DataFrame(out)
