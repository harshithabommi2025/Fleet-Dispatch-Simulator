"""Live demo: compare robotaxi dispatch policies in the browser.

    streamlit run streamlit_app.py
"""
import altair as alt
import pandas as pd
import streamlit as st

from fleetsim import POLICIES, SimConfig
from fleetsim.experiments import compare_to_baseline, confidence_intervals, run_experiment

st.set_page_config(page_title="Fleet Dispatch Simulator", layout="wide")
st.title("Fleet Dispatch Simulator")
st.markdown("Simulate an autonomous ride-hailing fleet and compare how **dispatch policies** "
            "affect rider wait times and fleet efficiency. Change the scenario on the left, then click **Run**.")

METRICS = {
    "mean_wait_s": "Mean rider wait (s)",
    "service_rate": "Service rate",
    "utilization": "Vehicle utilization",
    "deadhead_ratio": "Deadhead (empty-km) ratio",
    "trips_per_vehicle_hour": "Trips per vehicle-hour",
    "p90_wait_s": "P90 rider wait (s)",
}

with st.sidebar:
    st.header("Scenario")
    num_vehicles = st.slider("Fleet size", 20, 400, 150, step=10)
    demand = st.slider("Ride requests per hour", 100, 1500, 600, step=50)
    peak = st.slider("Rush-hour peak multiplier", 1.0, 4.0, 2.0, step=0.5)
    patience = st.slider("Rider patience (s)", 120, 1200, 600, step=60)
    hours = st.select_slider("Simulated hours", [1, 2, 4], value=2)
    st.header("Experiment")
    policies = st.multiselect("Dispatch policies", list(POLICIES), default=list(POLICIES),
                              help="\n".join(f"**{k}**: {v.__doc__.strip().splitlines()[0]}"
                                             for k, v in POLICIES.items()))
    baseline = st.selectbox("Baseline", list(POLICIES), index=list(POLICIES).index("greedy"))
    seeds = st.slider("Random seeds", 2, 10, 5, help="More seeds = tighter confidence intervals")
    run = st.button("Run", type="primary", width="stretch")


@st.cache_data(show_spinner=False)
def simulate(cfg: dict, policies: tuple, seeds: int) -> pd.DataFrame:
    return run_experiment(SimConfig(**cfg), policies, range(seeds))


if not policies:
    st.warning("Pick at least one policy.")
    st.stop()

cfg = dict(num_vehicles=num_vehicles, demand_rate_per_hour=float(demand), peak_multiplier=peak,
           patience_s=patience, horizon_s=hours * 3600)
if run or "df" not in st.session_state:
    with st.spinner(f"Running {len(policies) * seeds} simulations…"):
        st.session_state.df = simulate(cfg, tuple(policies), seeds)
df = st.session_state.df

# Headline numbers
if baseline in df.policy.values and df.policy.nunique() > 1:
    cmp = compare_to_baseline(df, baseline)
    best = cmp[cmp.metric == "mean_wait_s"].sort_values("pct_change").iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Best policy vs {baseline}", best["policy"])
    c2.metric("Mean rider wait", f"{best['policy_mean']:.0f} s", f"{best['pct_change']:+.1f}%", delta_color="inverse")
    util = cmp[(cmp.metric == "utilization") & (cmp.policy == best["policy"])].iloc[0]
    c3.metric("Vehicle utilization", f"{util['policy_mean']:.0%}", f"{util['pct_change']:+.1f}%")

# KPI charts
summary = confidence_intervals(df)
cols = st.columns(3)
for i, (metric, label) in enumerate(METRICS.items()):
    d = summary[summary.metric == metric]
    bars = alt.Chart(d).mark_bar().encode(
        x=alt.X("policy:N", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("mean:Q", title=None), color=alt.Color("policy:N", legend=None),
        tooltip=["policy", alt.Tooltip("mean:Q", format=".3f"),
                 alt.Tooltip("ci_low:Q", format=".3f"), alt.Tooltip("ci_high:Q", format=".3f")])
    err = alt.Chart(d).mark_errorbar(ticks=True).encode(x="policy:N", y=alt.Y("ci_low:Q", title=None),
                                                        y2="ci_high:Q")
    cols[i % 3].markdown(f"**{label}**")
    cols[i % 3].altair_chart((bars + err).properties(height=220), width="stretch")
st.caption("Bars: mean across seeds. Whiskers: 95% confidence interval. "
           "Every policy sees identical demand per seed (common random numbers).")

if baseline in df.policy.values and df.policy.nunique() > 1:
    st.subheader(f"Versus baseline: {baseline} (paired t-test)")
    show = cmp.assign(metric=cmp.metric.map(METRICS))
    colour = {"IMPROVED": "background-color:#d4edda", "REGRESSION": "background-color:#f8d7da"}
    st.dataframe(show.style.map(lambda v: colour.get(v, ""), subset=["verdict"])
                 .format({"baseline_mean": "{:.3f}", "policy_mean": "{:.3f}",
                          "pct_change": "{:+.1f}%", "p_value": "{:.4f}"}),
                 hide_index=True, width="stretch")

st.download_button("Download per-run CSV", df.to_csv(index=False), "fleetsim_results.csv", "text/csv")
st.markdown("Source code: [github.com/harshithabommi/fleet-dispatch-sim]"
            "(https://github.com/harshithabommi/fleet-dispatch-sim)")
