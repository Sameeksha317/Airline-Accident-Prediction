# Aviation Accident Analysis: Severity Prediction & Risk Profiling

End-to-end data science project built in **Dataiku**, applying the full AI/ML lifecycle to a 3,000-record aviation accident dataset; from data preparation through model deployment, to predict accident severity, predict survivor location, and cluster accidents into risk profiles - includes full video walkthrough.

## Business Context

Commercial aviation accident rates have fallen from 3.72 per million sectors (2005) to 1.13 (2024), yet 2024 still saw 244 on-board fatalities against a five-year average of 144 (IATA, 2024). General aviation accidents cost an estimated $1.64–$4.64 billion USD annually in medical costs, aircraft damage, legal claims, and investigations. This project explores how predictive analytics can support airlines and regulators in making data-driven safety decisions.

## Objectives

1. **Classify accident severity** (Minor vs. Severe) — identify which conditions (weather, aircraft type, pilot experience, altitude) are most predictive of a severe outcome.
2. **Predict majority survivor location** (Front vs. Back of aircraft) — to inform targeted emergency response planning.
3. **Cluster accidents by risk profile** — group accidents by shared characteristics to surface actionable safety insights.

## Dataset

`data/airline_accident_dataset.csv` — 3,000 records, 2010–2025, with 15 fields including airline, severity, fatalities/survivors, aircraft type, accident cause, location, weather, altitude, speed, pilot experience, time of day, survivor location, and plane damage. Full field definitions are in `docs/data_dictionary.docx`.

## Methodology

**Data preparation (Dataiku Prepare recipe):**
- Parsed and standardized `Accident_Date`, then extracted `Accident_Year` and `Accident_Month` to capture seasonality
- Dropped `Fatalities` and `Survivors` to avoid data leakage into the severity target
- Engineered `Speed_Altitude_Ratio` to capture the interaction between speed and altitude at the time of impact

**Train/test split:** 80/20 full random dispatch, seed 42, for reproducibility (2,400 train / 600 test rows)

**Modeling:** Decision Tree, Logistic Regression, and Random Forest were each trained and evaluated on identical held-out test data via Dataiku's Evaluate recipe (not training metrics), ensuring honest, out-of-sample performance reporting.

**Clustering:** K-Means tested at k=3, k=5, and k=7; k=7 selected on silhouette score (0.058).

## Key Findings

| Task | Best Model | Metric |
|---|---|---|
| Severity prediction | Logistic Regression | ROC AUC 0.570 |
| Survivor location prediction | Decision Tree | ROC AUC 0.518 |
| Risk clustering | K-Means (k=7) | Silhouette 0.058 |

- No single variable strongly predicts severity in isolation — pilot experience, weather, and accident cause each show limited individual influence, pointing to a multi-causal, multi-variable relationship.
- Both classification models exhibited class-imbalance bias (over-predicting the majority class); threshold optimization was applied in Dataiku (0.375 and 0.200 respectively) as a first-pass fix, with SMOTE proposed as a stronger future remedy.
- Clustering revealed three broad risk profiles differentiated mainly by **speed** and **pilot experience**: low-experience/low-speed, experienced/high-speed, and a mid-experience transitional group.

## Limitations

- The dataset's accident causes are near-uniformly distributed across all 12 categories, which is atypical of real-world aviation data (where pilot error and mechanical failure dominate) and likely limits generalizability.
- Key real-world predictors — impact angle, seat positioning, maintenance history — are not available in this dataset.
- Both target variables suffer from class imbalance, capping achievable model precision.

## Future Work

- Apply SMOTE via a Python recipe in Dataiku to rebalance classes before retraining
- Feed new accident records into the deployed models for real-time scoring
- Connect scored outputs to Power BI for operational safety dashboards and alerts

## Tools

Dataiku DSS (data preparation, modeling, evaluation, clustering) · Python · Statistical/ML methods: Logistic Regression, Decision Tree, Random Forest, K-Means

## Video Walkthrough

[![Watch the full walkthrough](https://img.youtube.com/vi/eSagg3uVORQ/hqdefault.jpg)](https://youtu.be/eSagg3uVORQ)

🎥 A slide-by-slide narration covering the business case, Dataiku pipeline, model results, and limitations.

## Repository Contents

- `data/airline_accident_dataset.csv` — source dataset (3,000 rows)
- `docs/data_dictionary.docx` — field definitions
- `presentation/Aviation_Accident_AI_Lifecycle_Sameeksha_Mathur.pptx` — full project write-up: domain research, methodology, visualizations, model results, limitations, and conclusions
- `presentation/Video_Narration_Script.docx` — narration script for the video walkthrough

## Author

**Sameeksha Mathur** — MBA, Business Analytics & AI, Middlesex University Dubai
Module: MSO4803 — AI Lifecycle
