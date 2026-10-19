# Architecture — Causal Marketing Intelligence

This document describes the intended architecture for the Marketing Incrementality & Uplift Optimization project. It is a research-oriented analytics application, not a production marketing platform. The design prioritizes reproducibility, statistically valid evaluation, and clear separation between model estimates and business decisions.

## 1. Architecture Goals

1. Make data preparation and analysis reproducible.
2. Separate treatment-effect estimation from policy selection and economic assumptions.
3. Prevent leakage between training, model selection, and final evaluation.
4. Compare advanced methods against simple, meaningful baselines.
5. Surface uncertainty and assumptions alongside business recommendations.
6. Keep the first implementation small enough to complete in 4–6 weeks.

## 2. High-Level Architecture

```text
                         +--------------------------+
                         | Public randomized dataset|
                         | Hillstrom email campaign |
                         +------------+-------------+
                                      |
                                      v
                         +--------------------------+
                         | Ingestion & data checks  |
                         | Schema / types / missing |
                         +------------+-------------+
                                      |
                                      v
                         +--------------------------+
                         | Feature & outcome setup  |
                         | X: pre-treatment features|
                         | T: treatment assignment  |
                         | Y: outcomes               |
                         +------------+-------------+
                                      |
                    +-----------------+------------------+
                    |                                    |
                    v                                    v
       +--------------------------+         +--------------------------+
       | Experimental analysis    |         | Train / validation / test|
       | ATE, intervals, balance  |         | reproducible split        |
       +------------+-------------+         +-------------+------------+
                    |                                     |
                    |                                     v
                    |                        +--------------------------+
                    |                        | Baseline policies/models |
                    |                        | random / broad / propensity|
                    |                        | T-learner                |
                    |                        +-------------+------------+
                    |                                      |
                    |                                      v
                    |                        +--------------------------+
                    |                        | Advanced uplift models   |
                    |                        | X-learner / causal forest|
                    |                        +-------------+------------+
                    |                                      |
                    +----------------------+---------------+
                                           |
                                           v
                              +--------------------------+
                              | Held-out policy evaluation|
                              | value / uncertainty / lift |
                              +------------+-------------+
                                           |
                                           v
                              +--------------------------+
                              | Budget & profit optimizer|
                              | costs / margin / capacity|
                              +------------+-------------+
                                           |
                           +---------------+----------------+
                           |                                |
                           v                                v
                +----------------------+         +----------------------+
                | Streamlit dashboard  |         | Research report      |
                | policy scenarios     |         | methods / limitations|
                +----------------------+         +----------------------+
```

## 3. Component Responsibilities

### 3.1 Data ingestion and validation

**Suggested module:** `src/data.py`

Responsibilities:

- Load the locally downloaded dataset.
- Verify required columns and expected data types.
- Check missingness, duplicates, invalid values, and treatment labels.
- Create a documented treatment indicator for one selected email campaign versus the no-email control.
- Keep raw data immutable and save processed artifacts separately if needed.

Validation rules:

- Treatment must have a documented meaning and valid values.
- Outcomes must match their definitions; conversion should be binary if encoded that way.
- Customer features must be measured before treatment.
- Post-treatment outcomes (`visit`, `conversion`, `spend`) must not be included in the feature matrix.
- Record data source, file version, and preprocessing choices.

### 3.2 Experimental and statistical analysis

**Suggested module:** `src/inference.py`

Responsibilities:

- Estimate ATE for conversion and spend.
- Produce confidence intervals and appropriate hypothesis-test results.
- Calculate pre-treatment standardized mean differences between treatment and control.
- Run sensitivity analyses, such as bootstrap intervals.
- Document the estimand, assumptions, sample sizes, and interpretation.

Primary estimand for outcome `Y`:

\[
ATE = E[Y(1)-Y(0)]
\]

The difference in randomized group means is the primary simple estimator. Regression adjustment may improve precision, but should not obscure the original experimental comparison.

### 3.3 Modeling

**Suggested module:** `src/models.py`

Inputs:

- `X`: pre-treatment customer features
- `T`: randomized treatment assignment
- `Y`: selected outcome

Candidate models:

- Propensity outcome model: predicts conversion, used for conventional targeting.
- T-learner: trains separate outcome models for treatment and control.
- X-learner: alternative meta-learner for heterogeneous treatment effects.
- Causal forest: estimates nonlinear treatment-effect heterogeneity.

The model API should return a documented estimate of treatment effect, plus uncertainty where the chosen method supports it. Do not imply that estimated conditional effects are observable individual-level ground truth.

### 3.4 Policy generation

**Suggested module:** `src/optimization.py`

A policy maps pre-treatment customer features to a treatment decision:

\[
\pi(X_i)\in\{0,1\}
\]

Initial policies:

- No campaign
- Random targeting at matched coverage
- Broad treatment of all eligible customers
- Top-k conversion propensity
- Top-k estimated uplift
- Positive expected incremental profit subject to budget

The policy should use only information available before treatment. Customer-level treatment-effect scores are model estimates, not directly observed causal effects.

### 3.5 Policy evaluation

**Suggested module:** `src/evaluation.py`

Responsibilities:

- Evaluate policies using held-out data.
- Use the known assignment mechanism when computing inverse-propensity-weighted estimates.
- Support doubly robust policy-value estimation if implemented correctly.
- Calculate confidence intervals using an appropriate method.
- Compare all policies at the same coverage or budget.
- Prevent model selection based on the final test set.

For a binary randomized treatment with known assignment probability \(e(X)\), an inverse-propensity-weighted policy-value estimator can be written as:

\[
\widehat V(\pi)=\frac{1}{n}\sum_{i=1}^{n}
\left[
\frac{T_i\pi(X_i)Y_i}{e(X_i)}
+
\frac{(1-T_i)(1-\pi(X_i))Y_i}{1-e(X_i)}
\right]
\]

This estimator evaluates the expected outcome under a policy that treats customers when \(\pi(X_i)=1\) and otherwise leaves them untreated. The assignment probabilities must be known or correctly estimated, and the data must support the policy being evaluated. Use the actual experiment design and treatment probabilities; do not assume a 50/50 allocation without verification.

### 3.6 Economic optimization

**Suggested module:** `src/optimization.py`

For conversion-based economics, one simple expected incremental profit score is:

\[
\widehat{\pi}_i=m\widehat{\tau}_i-c_i
\]

where `m` is contribution margin per incremental conversion, \(\widehat{\tau}_i\) is estimated incremental conversion probability, and `c_i` is treatment cost per customer.

Select actions \(a_i \in \{0,1\}\) to maximize:

\[
\max \sum_i a_i\widehat{\pi}_i
\quad \text{subject to} \quad
\sum_i a_i c_i \leq B
\]

where `B` is the campaign budget.

Important design notes:

- These economics are assumptions unless verified by the dataset or a business partner.
- If using spend as the outcome, model incremental spend consistently and subtract costs once.
- If costs differ, ranking customers by uplift alone may not maximize profit.
- Show sensitivity to contribution margin, campaign cost, and budget.
- Treat the optimized expected profit as a model-based estimate, then evaluate the selected policy on held-out data.

## 4. Data Flow and Leakage Controls

### Feature / treatment / outcome separation

| Variable group | Examples | Permitted use |
|---|---|---|
| Pre-treatment features `X` | Historical purchase behavior, recency, channel, customer status | Modeling and targeting |
| Treatment `T` | Email campaign assignment | Causal estimation and evaluation |
| Outcomes `Y` | Visit, conversion, spend | Training as target; evaluation |
| Economic assumptions | Contribution margin, campaign cost, budget | Scenario and optimization inputs |

Never use outcomes observed after campaign assignment as input features for the same campaign's targeting policy. Do not choose hyperparameters or policy thresholds based on final test outcomes.

### Evaluation separation

1. Create reproducible train, validation, and test partitions.
2. Fit feature transformations and nuisance models on training data only.
3. Use validation data for model and policy choices.
4. Freeze the chosen pipeline.
5. Evaluate final policies on the held-out test partition.
6. Report uncertainty and repeatability checks separately from the primary test result.

Since treatment was randomized, splitting should preserve usable treatment and control samples in each partition. Consider repeated splits or cross-fitting as robustness analyses, but keep the primary final evaluation clearly defined.

## 5. Dashboard Architecture

**Suggested entry point:** `app/dashboard.py`

The dashboard reads processed evaluation results and policy scores. It should not silently retrain models or alter the final test evaluation when a user changes economic assumptions.

### Planned views

1. **Experiment overview:** treatment counts, conversion rates, spend, ATE and intervals.
2. **Model comparison:** propensity and uplift ranking curves, model diagnostics.
3. **Policy comparison:** held-out value estimates, incremental outcome, uncertainty.
4. **Budget simulator:** budget, cost, margin, capacity, policy and sensitivity analysis.
5. **Segment analysis:** descriptive treatment-effect estimates with sample sizes and uncertainty.

Label outputs as one of:

- Observed experimental estimate
- Held-out policy-value estimate
- Model prediction
- User-supplied business assumption
- Simulated scenario

This prevents a scenario projection from being mistaken for realized business impact.

## 6. Repository Mapping

```text
data/raw/                  Original downloaded dataset (normally not committed)
notebooks/01_data_audit.ipynb
notebooks/02_experimental_analysis.ipynb
notebooks/03_uplift_models.ipynb
notebooks/04_policy_evaluation.ipynb
src/data.py                Ingestion and validation
src/inference.py           ATE and statistical analysis
src/models.py              Baseline and uplift estimators
src/evaluation.py          Ranking and policy evaluation
src/optimization.py        Budget allocation and economics
app/dashboard.py           Streamlit application
tests/                     Unit tests and data-contract tests
reports/                   Research report and executive summary
```

Keep core analysis in importable Python modules where practical; notebooks should explain experiments and findings rather than duplicate the entire codebase.

## 7. Testing and Reproducibility

Minimum checks:

- Data loader fails clearly when required columns are missing.
- Treatment labels are validated against the dataset.
- No outcome columns enter the feature matrix.
- Splits and model seeds are recorded.
- Economic calculations have unit tests and handle zero / invalid budgets.
- Optimization never exceeds the budget constraint.
- Policy evaluation uses only the designated evaluation partition.
- Notebook outputs can be regenerated from documented instructions.
- Reported results are generated from code, not manually typed into charts.

Track package versions in `requirements.txt`. Add a fixed configuration for the primary experiment, including treatment group, control group, outcome, split seed, budget assumptions, and selected evaluation metrics.

## 8. Security and Data Governance

- Do not commit credentials, tokens, or personal data.
- Check dataset license and terms before redistribution.
- Keep local raw files out of Git unless redistribution is permitted.
- Use synthetic or aggregated examples in public screenshots if needed.
- This is an offline research prototype; do not use it to contact customers or launch campaigns automatically.

## 9. Deployment Plan

For the portfolio scope:

1. Run analysis and model training locally.
2. Save model artifacts and evaluation outputs with version metadata.
3. Run the Streamlit dashboard locally.
4. Optionally deploy the dashboard using a compatible hosting service after checking data and dependency constraints.
5. Publish the research report, limitations, and reproducibility instructions in the repository.

A separate FastAPI service, database, orchestration platform, or cloud ML stack is not required for the initial version. Add these only if they support a clear demonstration goal.

## 10. Architecture Decision Summary

| Decision | Rationale |
|---|---|
| Python-first workflow | Keeps research iteration fast |
| Randomized campaign dataset | Supports a clearer causal interpretation |
| One primary treatment vs control | Keeps the study focused |
| Baselines before advanced models | Provides meaningful comparisons |
| Separate policy evaluation | Reduces optimistic performance estimates |
| Streamlit dashboard | Makes results accessible with minimal UI overhead |
| Explicit economic assumptions | Keeps business conclusions auditable |
| No automated campaign sending | Keeps the portfolio project safe and research-focused |

## 11. Known Limitations

- The dataset is historical and may not represent current marketing conditions.
- Uplift estimates can be noisy and subgroup estimates may have high variance.
- Randomized assignment supports causal estimation for the tested intervention and study population, not universal claims.
- Model-based policy value is not the same as a prospective production result.
- Profit depends on assumptions about margins, costs, and operational constraints.
- A real deployment should be validated with a prospective randomized experiment and appropriate monitoring.
