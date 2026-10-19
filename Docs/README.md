# Causal Marketing Intelligence: Incrementality & Uplift Optimization

A research-oriented Data Science project for estimating the incremental impact of marketing campaigns, identifying customers whose behavior may change because of an intervention, and allocating campaign budgets to maximize expected incremental profit.

> **Project status:** Research / development  
> **Primary use case:** Product and Business Data Science  
> **Planned timeline:** 4–6 weeks  
> **Dataset:** Hillstrom MineThatData E-Mail Analytics Challenge

## 1. Business Problem

Conventional propensity models estimate who is likely to purchase. That is not the same as estimating who will purchase *because of* a marketing campaign. Targeting customers who would buy anyway can waste marketing budget.

This project compares conventional targeting with causal uplift-based policies and evaluates whether those policies improve outcomes under a constrained budget.

### Research question

**Can customer-level treatment-effect estimates help a business allocate a limited marketing budget more effectively than random, broad, or purchase-propensity-based targeting?**

### Research questions

1. What is the average effect of an email campaign on conversion and customer spend?
2. Does uplift-based targeting outperform propensity-based targeting on held-out randomized data?
3. How reliably can heterogeneous treatment effects be estimated across customer segments?
4. Which targeting policy maximizes expected incremental profit under different budget and cost assumptions?
5. How sensitive are the conclusions to uncertainty, model choice, and economic assumptions?

## 2. Objectives

- Audit a public randomized marketing campaign dataset.
- Estimate average treatment effects (ATEs) with uncertainty intervals.
- Establish simple, reproducible targeting baselines.
- Train and compare uplift / heterogeneous treatment-effect models.
- Evaluate targeting policies on held-out data.
- Optimize campaign allocation under budget constraints.
- Deliver an interactive dashboard and a reproducible research report.

## 3. Dataset

**Hillstrom MineThatData E-Mail Analytics Challenge** — a historical customer-level email marketing dataset with treatment groups, including a no-email control group.

Find the dataset by searching for **“Hillstrom MineThatData email analytics dataset”** on a reputable public dataset repository or the original challenge materials.

Expected fields in common versions include:

- `segment`: campaign assignment / treatment group
- `visit`: website visit outcome
- `conversion`: purchase conversion outcome
- `spend`: customer spending outcome
- Customer attributes such as purchase history, recency, channel, and customer status

**Important:** Dataset mirrors may differ in filename, schema, labels, and packaging. Verify the downloaded data dictionary and exact treatment labels before running the notebooks. Do not assume a filename or label without checking.

### Data handling principles

- Keep the original data in `data/raw/`; do not commit it unless its license and repository policy permit it.
- Document the data source, version, and any preprocessing.
- Use only pre-treatment features to predict treatment effects.
- Do not use post-campaign outcomes such as `visit`, `conversion`, or `spend` as customer features.
- Preserve the control group and treatment assignment for valid evaluation.

## 4. Methodology

### 4.1 Data audit and exploratory analysis

- Validate schema, types, missing values, duplicates, ranges, and outcome distributions.
- Inspect treatment-group sizes and outcome rates.
- Examine skew and zero-inflation in spending.
- Check pre-treatment covariate balance using standardized mean differences.
- Document potential data quality and generalizability limitations.

### 4.2 Experimental analysis

For a selected email treatment versus the no-email control group, estimate the average treatment effect:

\[
\widehat{ATE} = \bar{Y}_{T} - \bar{Y}_{C}
\]

Analyze conversion and spend separately. Report effect sizes, confidence intervals, sample sizes, and p-values where appropriate. Use randomization-aware inference and bootstrap sensitivity analyses where useful.

### 4.3 Baselines

Compare the following policies:

1. No campaign / control.
2. Random targeting at the same campaign coverage.
3. Broad targeting of all eligible customers.
4. Purchase-propensity targeting.
5. T-learner treatment-effect model.

### 4.4 Advanced models

Add models only when they can be evaluated fairly against baselines:

- S-learner and X-learner
- Causal forest / generalized random forest
- Doubly robust treatment-effect estimation or policy evaluation

Candidate libraries include scikit-learn and EconML or CausalML. Verify package compatibility and method assumptions before selecting an implementation.

### 4.5 Policy evaluation

Use a fixed, reproducible train / validation / test split. Fit and tune models without using the final test set. Evaluate policies on held-out randomized observations using estimators appropriate to the assignment mechanism, such as inverse-propensity-weighted or doubly robust policy value estimates.

Metrics may include:

- Average treatment effect with confidence intervals
- Uplift / Qini curves and a clearly specified AUUC or Qini definition
- Incremental conversion at top-k targeting coverage
- Held-out policy value and uncertainty intervals
- Incremental spend and estimated incremental profit
- Sensitivity to costs, margin assumptions, and budget limits

Individual treatment effects are not directly observable. Model-estimated uplift should not be described as the true causal effect for a specific customer.

### 4.6 Budget optimization

For customer \(i\), a simple expected incremental profit score is:

\[
\widehat{\pi}_i = m\widehat{\tau}_i - c_i
\]

where:

- \(\widehat{\tau}_i\) is estimated incremental conversion probability
- \(m\) is assumed contribution margin per incremental conversion
- \(c_i\) is the cost of targeting customer \(i\)

For a budget \(B\), choose targeting decisions \(a_i \in \{0,1\}\):

\[
\max_{a_i} \sum_i a_i\widehat{\pi}_i
\quad \text{subject to} \quad
\sum_i a_i c_i \leq B
\]

This formulation is a starting point. If outcomes are modeled as spend, define the economic objective consistently and avoid double-counting revenue or costs. Report economic assumptions explicitly and include sensitivity analysis.

## 5. Planned Technology Stack

| Area | Tools |
|---|---|
| Data analysis | Python, Pandas, NumPy |
| Statistics | SciPy, statsmodels |
| Machine learning | scikit-learn (RandomForest = baseline, HistGradientBoosting = efficiency benchmark), XGBoost via its scikit-learn API (primary candidate) |
| Causal inference | EconML (`CausalForestDML`); S/T/X/DR-learners implemented in `src/models.py` on any of the three base learners |
| Optimization | SciPy |
| Visualization | Matplotlib, Plotly |
| Interactive app | Streamlit |
| SQL / local analytics | DuckDB (optional) |
| Testing | pytest |
| Version control | Git and GitHub |

The initial implementation should prioritize a reliable Python pipeline and Streamlit dashboard. A separate API or frontend is optional and should not displace the core research work.

## 6. Architecture Overview

```text
Public randomized campaign dataset
                |
                v
       Data ingestion & validation
                |
                v
       Exploratory / statistical analysis
                |
                v
   ATE estimation and baseline policies
                |
                v
  Uplift / heterogeneous-effect modeling
                |
                v
    Held-out policy evaluation
                |
                v
    Budget & profit optimization
                |
          +-----+------+
          |            |
          v            v
  Streamlit dashboard  Research report
```

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for component responsibilities, data flow, validation strategy, and key design decisions.

## 7. Planned Repository Structure

```text
causal-marketing-optimizer/
├── data/
│   └── raw/                     # Local data; usually git-ignored
├── notebooks/
│   ├── 01_data_understanding_experimental_analysis.ipynb
│   ├── 02_baseline_models.ipynb
│   ├── 03_advanced_uplift_models.ipynb
│   ├── 04_policy_evaluation_optimization.ipynb
│   ├── 05_interactive_dashboard.ipynb
│   ├── 06_final_evaluation_reproducibility.ipynb
│   └── 07_base_learner_study.ipynb
├── src/
│   ├── config.py
│   ├── data.py
│   ├── inference.py
│   ├── models.py
│   ├── evaluation.py
│   ├── optimization.py
│   ├── pipeline.py
│   ├── scenario.py
│   ├── benchmark.py
│   └── plotting.py
├── app/
│   └── dashboard.py
├── tests/
├── reports/
│   ├── research_report.md
│   └── executive_summary.md
├── requirements.txt
├── README.md
└── ARCHITECTURE.md
```

This is the implemented structure (updated from the original plan). See the root `README.md` for results.

## 8. Getting Started

### Prerequisites

- Python 3.11 or 3.12 recommended; confirm compatibility of causal inference libraries for your environment.
- Git
- A downloaded copy of the dataset

### Windows PowerShell setup

```powershell
git clone <your-repository-url>
cd causal-marketing-optimizer

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install pandas numpy scipy matplotlib seaborn scikit-learn statsmodels
pip install jupyter ipykernel plotly streamlit duckdb
pip install econml xgboost

pip freeze > requirements.txt
```

If installation of an optional causal library fails, install and validate it separately rather than blocking data auditing and baseline analysis.

### Add the dataset

Place the downloaded CSV in `data/raw/`. Update notebook paths and treatment labels after verifying the exact file schema.

### Run notebooks

```powershell
jupyter notebook
```

Start with `notebooks/01_data_audit.ipynb` and proceed in sequence. Do not build the final dashboard until the statistical analysis and evaluation pipeline are reproducible.

### Run the dashboard (once implemented)

```powershell
streamlit run app/dashboard.py
```

## 9. Milestones

| Period | Milestone | Expected deliverable |
|---|---|---|
| Week 1 | Data audit and experiment analysis | Validated dataset, ATE estimates, balance analysis |
| Week 2 | Baselines | Propensity model and simple targeting-policy comparisons |
| Week 3 | Advanced uplift models | X-learner / causal forest comparison and uncertainty analysis |
| Week 4 | Policy evaluation and optimization | Held-out policy values and budget scenarios |
| Week 5 | Dashboard | Interactive campaign and budget decision tool |
| Week 6 | Finalization | Reproducible report, tests, README, and demo |

For a four-week schedule, prioritize one treatment arm, core baselines, one or two advanced models, held-out policy evaluation, and a minimal dashboard.

## 10. Definition of Done

- [ ] Data source and license / terms are documented.
- [ ] Treatment and control labels are verified from the actual dataset.
- [ ] Pre-treatment features are separated from post-treatment outcomes.
- [ ] ATE estimates include uncertainty intervals.
- [ ] Propensity targeting and random / broad baselines are implemented.
- [ ] At least one advanced uplift model is evaluated.
- [ ] Model selection and final policy evaluation use separate data.
- [ ] Policy values and business economics are reported with assumptions.
- [ ] Results are reproducible from documented instructions.
- [ ] Dashboard and research report distinguish measured findings from simulations.

## 11. Business Value

The project is designed to help answer:

- Does a campaign generate incremental conversions or spend?
- Which targeting policy creates the greatest expected value?
- How does the recommended policy change under different budget and cost assumptions?
- How uncertain are the estimated gains?
- What evidence would be required before deploying the policy in a live campaign?

Business impact must be reported from actual held-out analysis or a clearly labeled simulation. Do not claim revenue increases, savings, or model performance before measuring them.

## 12. Limitations and Responsible Interpretation

- Historical campaign results may not generalize to current customers, channels, or markets.
- Heterogeneous treatment-effect estimates can be noisy, especially for small segments.
- The experiment estimates the effect of the assigned campaign under the conditions in the dataset, not every possible marketing intervention.
- Profit optimization depends on contribution margin, campaign cost, and other assumptions that must be documented.
- A retrospective portfolio analysis does not replace a prospective randomized experiment before production rollout.

## 13. Portfolio Presentation

Present the work as a decision science project, not merely a model benchmark. The final report should include:

1. Business problem and research hypotheses
2. Dataset and experimental design
3. Statistical analysis and uncertainty
4. Baselines and advanced model comparison
5. Held-out policy evaluation
6. Budget optimization and sensitivity analysis
7. Limitations, recommendations, and next steps

**Suggested project title:** *Causal Marketing Intelligence: Incrementality, Uplift Modeling & Budget Optimization*
