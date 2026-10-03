import pytest

from fleetsim import SimConfig, Simulation, summarize
from fleetsim.demand import Request
from fleetsim.experiments import compare_to_baseline, run_experiment

SMALL = dict(num_vehicles=10, horizon_s=1800, drain_s=1800, demand_rate_per_hour=120)


def test_single_trip_timing():
    cfg = SimConfig(num_vehicles=1, speed_kmh=36, horizon_s=100, drain_s=1000)  # 36 km/h = 100 s per km
    req = Request(0, 0.0, origin=(1.0, 0.0), dest=(3.0, 0.0))
    sim = Simulation(cfg, "greedy", requests=[req])
    sim.vehicles[0].pos = (0.0, 0.0)
    sim.run()
    assert req.status == "served"
    assert req.wait_s == pytest.approx(100)          # 1 km empty
    assert req.dropped_at == pytest.approx(300)      # + 2 km with rider
    m = summarize(sim)
    assert m["deadhead_ratio"] == pytest.approx(1 / 3)


def test_rider_cancels_when_no_car():
    cfg = SimConfig(num_vehicles=1, speed_kmh=10, patience_s=60, max_pickup_s=30,
                    horizon_s=100, drain_s=200)
    req = Request(0, 0.0, origin=(9.0, 9.0), dest=(1.0, 1.0))
    sim = Simulation(cfg, "greedy", requests=[req])
    sim.vehicles[0].pos = (0.0, 0.0)
    sim.run()
    assert req.status == "cancelled"


@pytest.mark.parametrize("policy", ["random", "greedy", "batched"])
def test_policies_conserve_requests(policy):
    sim = Simulation(SimConfig(**SMALL), policy, seed=1).run()
    m = summarize(sim)
    assert m["served"] + m["cancelled"] == m["requests"] > 0
    assert 0 <= m["utilization"] <= 1


def test_same_seed_same_demand_across_policies():
    cfg = SimConfig(**SMALL)
    a = Simulation(cfg, "greedy", seed=3)
    b = Simulation(cfg, "batched", seed=3)
    assert [r.origin for r in a.requests] == [r.origin for r in b.requests]


def test_compare_flags_random_as_worse_on_wait():
    df = run_experiment(SimConfig(**SMALL), ["greedy", "random"], range(5))
    cmp = compare_to_baseline(df, "greedy", metrics=["mean_wait_s"])
    assert cmp.iloc[0]["pct_change"] > 0  # random waits longer than greedy
