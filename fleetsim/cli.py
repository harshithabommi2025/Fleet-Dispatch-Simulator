"""Command-line interface.

    python -m fleetsim run --policy batched
    python -m fleetsim compare --policies random greedy batched --seeds 10 --baseline greedy
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from .config import SimConfig
from .experiments import compare_to_baseline, confidence_intervals, run_experiment, run_once
from .policies import POLICIES


def _cfg(args) -> SimConfig:
    cfg = SimConfig.from_yaml(args.config) if args.config else SimConfig()
    for kv in args.set or []:
        k, v = kv.split("=", 1)
        cur = getattr(cfg, k)
        setattr(cfg, k, type(cur)(v) if cur is not None else v)
    return cfg


def main(argv=None):
    p = argparse.ArgumentParser(prog="fleetsim", description="Robotaxi fleet dispatch simulator")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="YAML config file")
    common.add_argument("--set", nargs="*", metavar="KEY=VALUE", help="override config values")

    r = sub.add_parser("run", parents=[common], help="run one simulation")
    r.add_argument("--policy", default="greedy", choices=sorted(POLICIES))
    r.add_argument("--seed", type=int, default=0)

    c = sub.add_parser("compare", parents=[common], help="compare policies over many seeds")
    c.add_argument("--policies", nargs="+", default=sorted(POLICIES))
    c.add_argument("--seeds", type=int, default=10)
    c.add_argument("--baseline", default="greedy")
    c.add_argument("--out", default="results.csv", help="per-run results CSV")

    sub.add_parser("policies", help="list available policies")
    args = p.parse_args(argv)

    if args.cmd == "policies":
        for name, cls in POLICIES.items():
            print(f"{name:10s} {cls.__doc__.strip().splitlines()[0]}")
        return

    cfg = _cfg(args)
    if args.cmd == "run":
        print(json.dumps(run_once(cfg, args.policy, args.seed), indent=2, default=float))
        return

    df = run_experiment(cfg, args.policies, range(args.seeds))
    df.to_csv(args.out, index=False)
    pd.set_option("display.width", 140)
    print("\n=== Mean with 95% confidence interval ===")
    ci = confidence_intervals(df)
    print(ci.pivot(index="metric", columns="policy", values="mean").round(3))
    if args.baseline in args.policies and len(args.policies) > 1:
        print(f"\n=== Versus baseline '{args.baseline}' (paired t-test) ===")
        cmp = compare_to_baseline(df, args.baseline)
        print(cmp.round({"baseline_mean": 3, "policy_mean": 3, "pct_change": 1, "p_value": 4})
                 .to_string(index=False))
    print(f"\nPer-run results written to {args.out}")
