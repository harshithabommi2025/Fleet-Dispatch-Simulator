# Fleet Dispatch Simulator

A discrete-event simulator for an autonomous ride-hailing (robotaxi) fleet. It lets you plug in different
**dispatch policies**, deciding which vehicle serves which rider, and measure how each one affects
**fleet efficiency**, with statistically sound comparisons across many random seeds.

## Features
- **SimPy discrete-event engine**: riders request trips, vehicles drive to pickups and drop-offs, and riders cancel if they wait too long.
- **Pluggable dispatch policies**:
  - `random`: baseline
  - `greedy`: oldest request gets the nearest car
  - `batched`: collects requests for 30 s, then solves a min-cost bipartite matching (Hungarian algorithm)
- **Realistic demand**: Poisson arrivals with a rush-hour peak and spatial hotspots, or real NYC taxi trips.
- **Fleet KPIs**: service rate, mean and p90 rider wait, vehicle utilization, deadhead (empty-mile) ratio, and trips per vehicle-hour.
- **Experiment framework**: multi-seed runs with common random numbers, 95% confidence intervals, a paired t-test against a baseline, and automatic **REGRESSION** flags.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

python -m fleetsim policies
python -m fleetsim run --policy batched
python -m fleetsim compare --policies random greedy batched --seeds 10 --baseline greedy
python -m fleetsim compare --set num_vehicles=150 demand_rate_per_hour=800   # what-if scenario
pytest
```

## Using real NYC taxi data
Download a 2015 NYC TLC yellow-taxi CSV (those files include GPS coordinates), then run:
```bash
python scripts/prepare_nyc_data.py yellow_tripdata_2015-01.csv --date 2015-01-15 --start 17:00 --hours 2 --sample 0.2
python -m fleetsim compare --config configs/nyc.yaml --seeds 5
```

## Adding your own policy
```python
from fleetsim.policies import DispatchPolicy, POLICIES

class MyPolicy(DispatchPolicy):
    """One-line description."""
    name = "mine"
    batched = True
    def assign(self, requests, vehicles, now, travel_time):
        return [...]   # list of (request, vehicle)

POLICIES["mine"] = MyPolicy
```

## Project layout
```
fleetsim/
  config.py       SimConfig dataclass (YAML-loadable)
  demand.py       synthetic and CSV demand
  policies.py     dispatch policies
  simulator.py    SimPy simulation
  metrics.py      fleet KPIs
  experiments.py  multi-seed runs, CIs, baseline comparison
  cli.py          command-line interface
```

## Modeling assumptions
- Manhattan-distance travel at a constant average speed.
- No repositioning of idle vehicles.
- Riders cancel after `patience_s` seconds without an assigned car.
