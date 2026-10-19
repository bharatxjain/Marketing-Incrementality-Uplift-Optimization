"""Campaign decision dashboard.

Run:  streamlit run app/dashboard.py

Reads the frozen held-out artifacts written by notebook 04. Changing an
assumption re-runs the allocation on fixed test-set scores; no model is retrained
and no test outcome is reused for fitting.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config, scenario
from src import plotting as P

st.set_page_config(page_title="Campaign Uplift Optimizer", page_icon="📬", layout="wide")

ART = config.ARTIFACTS_DIR
if not (ART / "test_scores.csv").exists():
    st.error("Artifacts not found. Run notebooks 01-04 first (see README).")
    st.stop()


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
@st.cache_data
def load():
    frame, meta = scenario.load_artifacts(ART)
    ate = pd.read_csv(ART / "ate_table.csv")
    curves = pd.read_csv(ART / "test_gain_curves.csv")
    ranking = pd.read_csv(ART / "test_ranking_metrics.csv")
    return frame, meta, ate, curves, ranking


frame, meta, ate_table, curves, ranking = load()
SELECTED, CHALLENGER = meta["selected_model"], meta["challenger_model"]
_results = ROOT / "reports" / "results.json"
RECOMMENDED = json.loads(_results.read_text()).get("recommended_model", SELECTED) if _results.exists() else SELECTED


@st.cache_data
def cached_scenario(option, margin, cost, budget, population):
    return scenario.run_scenario(frame, meta, option, margin, cost, budget, population)


@st.cache_data
def cached_budget_curve(option, margin, cost, population, max_budget):
    budgets = np.linspace(0, max_budget, 21)
    return scenario.budget_curve(frame, meta, option, margin, cost, budgets, population)


@st.cache_data
def cached_sensitivity(option, model, budget, population):
    margins = [0.15, 0.25, 0.35, 0.45, 0.55]
    costs = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
    out = np.zeros((len(margins), len(costs)))
    for i, m in enumerate(margins):
        for j, c in enumerate(costs):
            r = scenario.run_scenario(frame, meta, option, m, c, budget, population)
            out[i, j] = r.loc[model, "profit"] - r.drop(index=meta["uplift_models"]).profit.max()
    return margins, costs, out


@st.cache_data
def cached_segments(variable, model, option, margin, cost, budget, population):
    return scenario.segment_view(frame, meta, variable, model, option, margin, cost, budget, population)


def money(v, decimals=0):
    return ("−" if v < 0 else "") + f"${abs(v):,.{decimals}f}"


def tag(kind):
    st.caption(f"🏷️ {scenario.LABELS[kind]}")


def style(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), hovermode="closest",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                      font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif"))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(137,135,129,0.25)", zeroline=True, zerolinecolor="rgba(137,135,129,0.6)")
    return fig


# ----------------------------------------------------------------------------
# Sidebar: assumptions
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Scenario")
    option = st.radio("Campaign", list(scenario.CAMPAIGN_OPTIONS), index=0,
                      format_func=scenario.CAMPAIGN_OPTIONS.get)
    model_opts = meta["uplift_models"][::-1]
    tags = {RECOMMENDED: " (recommended)", SELECTED: " (pre-registered pick)"}
    model = st.selectbox("Uplift model", model_opts, index=model_opts.index(RECOMMENDED),
                         format_func=lambda m: m + tags.get(m, ""),
                         help=f"Pre-registered selection (dev AUGC): {SELECTED}. Recommended after the final "
                              f"evaluation, using a budgeted-profit criterion on development data: {RECOMMENDED}. See notebook 06.")
    st.divider()
    st.subheader("Business assumptions")
    st.caption("User-supplied, not estimated from the data")
    margin = st.slider("Contribution margin on incremental spend", 0.05, 0.80, float(config.MARGIN), 0.05, format="%.2f")
    cost = st.slider("Cost per e-mail ($)", 0.01, 0.50, float(config.COST_PER_CONTACT), 0.01, format="$%.2f")
    population = st.number_input("Customers in mailing universe", 10_000, 10_000_000, config.POPULATION_SIZE, 10_000)
    unlimited = st.checkbox("No budget limit", value=False)
    max_b = float(cost * population)
    budget = None if unlimited else st.slider(
        "Campaign budget ($)", 0.0, max_b, min(float(config.DEFAULT_BUDGET) * population / config.POPULATION_SIZE, max_b),
        max(max_b / 100, 1.0), format="$%.0f")
    if budget is not None:
        st.caption(f"Enough to e-mail {min(1, budget / max_b):.0%} of customers")

res = cached_scenario(option, margin, cost, budget, int(population))

st.title("Campaign Uplift Optimizer")
st.markdown(
    "Who should receive which e-mail, and how many e-mails are worth sending? Built on the randomized Hillstrom "
    "experiment (64,000 customers). Policy values are **held-out AIPW estimates** on 12,800 test customers "
    "the models never saw.")

tabs = st.tabs(["Policy comparison", "Budget scenarios", "Segment analysis", "Experiment", "Model diagnostics", "About"])

# ----------------------------------------------------------------------------
# 1. Policy comparison
# ----------------------------------------------------------------------------
with tabs[0]:
    best_base = res.drop(index=meta["uplift_models"]).profit.idxmax()
    c1, c2, c3, c4 = st.columns(4)
    r = res.loc[model]
    c1.metric(f"{model}: incremental profit", money(r.profit),
              f"{money(r.profit - res.loc[best_base, 'profit'])} vs best baseline ({best_base})")
    c2.metric("Customers e-mailed", f"{r.contacts:,.0f}", f"{r.coverage:.0%} of universe", delta_color="off")
    c3.metric("E-mail spend", money(r.spend_on_email), f"ROI {r.roi:.2f}" if np.isfinite(r.roi) else "", delta_color="off")
    c4.metric("Men's / Women's mix", f"{r.mens_share / max(r.coverage, 1e-9):.0%} / {1 - r.mens_share / max(r.coverage, 1e-9):.0%}"
              if r.coverage > 0 else "—")
    tag("heldout")

    order = ["No campaign", "Random"] + meta["ranking_models"] + meta["uplift_models"]
    rr = res.loc[order]
    fig = go.Figure(go.Bar(
        x=rr.profit, y=rr.index, orientation="h",
        marker_color=[P.color_for(p) if p != "No campaign" else P.AXIS for p in rr.index],
        error_x=dict(type="data", symmetric=False, array=rr.profit_ci_high - rr.profit,
                     arrayminus=rr.profit - rr.profit_ci_low, color=P.INK_2, thickness=1.2),
        customdata=np.column_stack([rr.coverage, rr.profit_ci_low, rr.profit_ci_high, rr.contacts]),
        hovertemplate="<b>%{y}</b><br>Profit %{x:$,.0f}<br>95% CI %{customdata[1]:$,.0f} to %{customdata[2]:$,.0f}"
                      "<br>Coverage %{customdata[0]:.0%} (%{customdata[3]:,.0f} e-mails)<extra></extra>"))
    fig.update_layout(title="Held-out incremental profit by policy (95% CI)", xaxis_tickprefix="$", xaxis_tickformat=",.0f")
    st.plotly_chart(style(fig, 360), width="stretch")

    tbl = rr[["kind", "coverage", "contacts", "spend_on_email", "inc_spend", "profit", "profit_ci_low", "profit_ci_high", "roi"]].copy()
    st.dataframe(tbl.style.format({"coverage": "{:.0%}", "contacts": "{:,.0f}", "spend_on_email": money, "inc_spend": money,
                                   "profit": money, "profit_ci_low": money, "profit_ci_high": money, "roi": "{:.2f}"},
                                  na_rep="—"), width="stretch")
    st.warning("**Read the intervals.** One 12,800-customer test set cannot separate most policies: intervals are "
               "about ±$10k per 100k customers. Pooled evaluation over all 64,000 customers (notebook 06) is more precise. "
               "Use this page to understand *how* each policy decides, not to rank close policies.")
    st.info("**How policies spend the budget.** Random and propensity rankings fill the whole budget on the campaign with the "
            "larger average effect: a ranking cannot tell whether an e-mail pays for itself. Uplift policies e-mail a customer "
            "only when margin × predicted incremental spend exceeds the cost, and pick the better campaign per customer.")

# ----------------------------------------------------------------------------
# 2. Budget scenarios
# ----------------------------------------------------------------------------
with tabs[1]:
    bc = cached_budget_curve(option, margin, cost, int(population), float(cost * population))
    fig = go.Figure()
    for name in ["Random"] + meta["ranking_models"] + meta["uplift_models"]:
        s = bc[bc.policy == name]
        if name == model:
            fig.add_trace(go.Scatter(x=np.r_[s.budget, s.budget[::-1]], y=np.r_[s.profit_ci_high, s.profit_ci_low[::-1]],
                                     fill="toself", fillcolor="rgba(0,131,0,0.12)", line=dict(width=0),
                                     hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=s.budget, y=s.profit, mode="lines+markers", name=name,
                                 line=dict(color=P.color_for(name), width=2.5 if name == model else 1.8),
                                 marker=dict(size=5),
                                 hovertemplate=f"<b>{name}</b><br>Budget %{{x:$,.0f}}<br>Profit %{{y:$,.0f}}<extra></extra>"))
    if budget is not None:
        fig.add_vline(x=budget, line_color=P.INK_2, line_width=1)
    fig.update_layout(title="Held-out profit as the budget grows", xaxis_title="Budget", yaxis_title="Incremental profit",
                      xaxis_tickprefix="$", yaxis_tickprefix="$", xaxis_tickformat=",.0f", yaxis_tickformat=",.0f",
                      hovermode="x unified")
    st.plotly_chart(style(fig, 420), width="stretch")
    tag("simulated")
    st.caption("Each point re-runs the allocation at that budget on the frozen test-set scores, then values it with "
               "held-out AIPW scores. Uplift curves flatten once every customer worth e-mailing has been reached.")

    st.subheader("Sensitivity to margin and cost")
    margins, costs, adv = cached_sensitivity(option, model, budget, int(population))
    lim = float(np.abs(adv).max()) or 1.0
    fig = go.Figure(go.Heatmap(
        z=adv, x=[f"${c:.2f}" for c in costs], y=[f"{m:.0%}" for m in margins], zmid=0, zmin=-lim, zmax=lim,
        colorscale=[[0, "#a02524"], [0.5, "#f0efec"], [1, "#184f95"]],
        text=[[money(v) for v in row] for row in adv], texttemplate="%{text}",
        hovertemplate="Margin %{y}, cost %{x}<br>Advantage %{text}<extra></extra>", colorbar=dict(title="$")))
    fig.update_layout(title=f"{model}: profit advantage over the best non-uplift policy",
                      xaxis_title="Cost per e-mail", yaxis_title="Margin")
    st.plotly_chart(style(fig, 360), width="stretch")
    tag("simulated")

# ----------------------------------------------------------------------------
# 3. Segment analysis
# ----------------------------------------------------------------------------
with tabs[2]:
    var_labels = {"merch_history": "Past merchandise", "channel": "Purchase channel", "customer_type": "New vs existing",
                  "zip_code": "Area", "recency_bucket": "Recency", "history_segment": "Past-year spend"}
    var = st.selectbox("Segment by", list(var_labels), format_func=var_labels.get)
    seg = cached_segments(var, model, option, margin, cost, budget, int(population))
    if var == "recency_bucket":
        seg = seg.set_index("segment").reindex(["1-3 mo", "4-6 mo", "7-9 mo", "10-12 mo"]).reset_index()
    fig = go.Figure()
    for a, label in [("mens", "Mens E-Mail"), ("womens", "Womens E-Mail")]:
        fig.add_trace(go.Scatter(
            x=seg[f"observed_{a}"], y=seg["segment"], mode="markers", name=f"{label}: observed",
            marker=dict(color=P.CAMPAIGN_COLORS[label], size=10, line=dict(color="white", width=1.5)),
            error_x=dict(type="data", symmetric=False, array=seg[f"hi_{a}"] - seg[f"observed_{a}"],
                         arrayminus=seg[f"observed_{a}"] - seg[f"lo_{a}"], thickness=1.5),
            customdata=np.column_stack([seg[f"lo_{a}"], seg[f"hi_{a}"], seg["customers"], seg[f"predicted_{a}"]]),
            hovertemplate="<b>%{y}</b><br>Observed %{x:$.2f} (95% CI %{customdata[0]:$.2f} to %{customdata[1]:$.2f})"
                          "<br>Predicted %{customdata[3]:$.2f}<br>n = %{customdata[2]:,}<extra>" + label + "</extra>"))
        fig.add_trace(go.Scatter(x=seg[f"predicted_{a}"], y=seg["segment"], mode="markers", name=f"{label}: {model}",
                                 marker=dict(color=P.CAMPAIGN_COLORS[label], size=9, symbol="diamond-open", line=dict(width=2)),
                                 hovertemplate="<b>%{y}</b><br>Predicted %{x:$.2f}<extra>" + label + "</extra>"))
    fig.add_vline(x=0, line_color=P.AXIS)
    fig.update_layout(title="Incremental spend per customer by segment", xaxis_title="Incremental spend per customer",
                      xaxis_tickprefix="$")
    st.plotly_chart(style(fig, 420), width="stretch")
    st.caption("🏷️ Dots: observed experimental estimate (held-out AIPW, 95% CI) · Diamonds: model prediction")

    show = seg[["segment", "customers", "observed_mens", "observed_womens", "predicted_mens", "predicted_womens",
                "share_targeted", "share_mens", "share_womens"]]
    st.dataframe(show.style.format({"customers": "{:,}", "observed_mens": "${:.2f}", "observed_womens": "${:.2f}",
                                    "predicted_mens": "${:.2f}", "predicted_womens": "${:.2f}", "share_targeted": "{:.0%}",
                                    "share_mens": "{:.0%}", "share_womens": "{:.0%}"}), width="stretch", hide_index=True)
    st.warning("Segment estimates on the test set use a few hundred to a few thousand customers each. Intervals are wide, "
               "and many segments are compared at once, so treat single segments as hypotheses, not findings.")

# ----------------------------------------------------------------------------
# 4. Experiment overview
# ----------------------------------------------------------------------------
with tabs[3]:
    sp = ate_table[ate_table.outcome == "spend"].set_index("campaign")
    c1, c2, c3 = st.columns(3)
    c1.metric("Customers randomized", "64,000", "3 arms of ~21.3k", delta_color="off")
    for col, label in [(c2, "Mens E-Mail"), (c3, "Womens E-Mail")]:
        r = sp.loc[label]
        col.metric(f"{label}: spend effect per customer", f"${r.ATE:.2f}",
                   f"95% CI ${r.ci_low:.2f}–${r.ci_high:.2f} · {r.rel_lift:+.0%}", delta_color="off")
    tag("observed")
    t = ate_table.copy()
    t["effect"] = np.where(t.outcome == "spend", t.ATE.map(lambda v: f"${v:.3f}"), (t.ATE * 100).map(lambda v: f"{v:+.2f} pp"))
    t["95% CI"] = np.where(t.outcome == "spend", t.apply(lambda r: f"${r.ci_low:.3f} to ${r.ci_high:.3f}", axis=1),
                           t.apply(lambda r: f"{r.ci_low * 100:+.2f} to {r.ci_high * 100:+.2f} pp", axis=1))
    t["relative lift"] = t.rel_lift.map("{:+.0%}".format)
    t["Holm p"] = t.p_holm.map("{:.1e}".format)
    st.dataframe(t[["campaign", "outcome", "effect", "95% CI", "relative lift", "Holm p"]], hide_index=True,
                 width="stretch")
    st.markdown("Randomization checks: sample-ratio test p = 0.90; all standardized mean differences < 0.02; "
                "a classifier cannot predict assignment from customer features (AUC ≈ 0.50).")

# ----------------------------------------------------------------------------
# 5. Model diagnostics
# ----------------------------------------------------------------------------
with tabs[4]:
    camp = "mens" if option == "mens" else "womens" if option == "womens" else st.radio(
        "Campaign", ["womens", "mens"], horizontal=True, format_func=lambda k: config.CAMPAIGNS[k])
    fig = go.Figure()
    for name in ["Random", "Purchase propensity", "T-learner", CHALLENGER, SELECTED]:
        s = curves[(curves.campaign == camp) & (curves.policy == name)]
        fig.add_trace(go.Scatter(x=s.coverage, y=s.gain, name=name, mode="lines",
                                 line=dict(color=P.color_for(name), width=2.4 if name == SELECTED else 1.7),
                                 hovertemplate=f"<b>{name}</b><br>Top %{{x:.0%}} e-mailed<br>%{{y:$.3f}} per customer<extra></extra>"))
    fig.update_layout(title=f"{config.CAMPAIGNS[camp]}: test-set gain curves (incremental spend per population customer)",
                      xaxis_tickformat=".0%", yaxis_tickprefix="$", xaxis_title="Share e-mailed", hovermode="x unified")
    st.plotly_chart(style(fig, 400), width="stretch")
    tag("heldout")
    rk = ranking[ranking.campaign == camp].set_index("model")[["AUGC", "gain@30%", "vs_random", "vs_random_se", "BLP_slope", "Qini_coef"]]
    st.dataframe(rk.style.format("{:.3f}"), width="stretch")
    st.caption("AUGC: area between gain curve and random ($/customer). BLP slope: held-out calibration of dollar "
               "predictions (1 = well scaled, 0 = noise; not meaningful for the propensity score).")

# ----------------------------------------------------------------------------
# 6. About
# ----------------------------------------------------------------------------
with tabs[5]:
    st.markdown(f"""
**Data.** Kevin Hillstrom's MineThatData E-Mail Analytics Challenge (2008): 64,000 customers randomized to a Men's
e-mail, a Women's e-mail, or no e-mail, with visits, conversions and spend tracked for two weeks.

**Method.** Models were trained on an 80% development set and selected by 5-fold cross-fitting
(pre-registered pick: **{SELECTED}**, challenger: **{CHALLENGER}**; recommended after final evaluation: **{RECOMMENDED}**). They were then refit once and scored a 20% test set. Every
number on this page is computed on that test set with doubly robust (AIPW) scores, using the known 1/3 assignment
probabilities.

**What changes when you move a slider.** Only the allocation rule is re-run: who is e-mailed, and with which campaign.
Models, scores and evaluation data stay fixed.

**Limitations.** This is a 2008 experiment, the economics are assumptions, and policy values are estimates with the
intervals shown. A real rollout should be validated with a fresh randomized holdout.
""")
