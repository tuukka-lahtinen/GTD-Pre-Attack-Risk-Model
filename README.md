# Predicting Terrorist Attack Severity from Target, Context, and Anticipated Tactical Features

A pre-attack risk model using the Global Terrorism Database (GTD). Course project for *Data Science for Business* (Aalto University). This is **Stage 1** — problem formulation and methodology; model training and test-set evaluation follow in Stage 2.

## Problem

Intelligence agencies often identify attacks in their planning stage, forming estimates of the intended target, method, and weapon before the attack occurs. This project asks how well expected attack severity can be predicted from such pre-attack assessments — supporting resource prioritization and emergency response planning based on anticipated risk rather than post-hoc reaction.

Formulated as supervised regression: given a feature vector describing an attack's target, context, and anticipated tactical characteristics, predict a continuous severity label.

## Data

- **Source:** [Global Terrorism Database (GTD)](https://www.start.umd.edu/gtd), maintained by START, University of Maryland
- **Scope:** 2000–2017, for consistent data collection standards and a contemporary threat landscape
- **After cleaning:** 103,387 attacks retained (92.4% of 111,855 raw attacks in this period)

## Target & features

- **Label:** `log(nkill + nwound + 1)` — log-transformed total casualties. Chosen over fatalities alone since many attacks cause injuries without deaths; the raw count is highly skewed, which the log transform corrects.
- **Target & context features:** `targtype1_grouped`, `region_txt`, `multiple`, `ishostkid`
- **Anticipated tactical features:** `attacktype1_txt`, `weaptype1_grouped`

**Key assumption:** tactical features (attack method, weapon) are treated as estimable via intelligence prior to the attack, not as data only confirmed afterward. This is explicit and untested — real-world performance would depend on the reliability of that intelligence. Purely post-hoc/outcome variables (perpetrators captured, claims of responsibility, damage assessments) are excluded by design to prevent leakage.

## Preprocessing

- Rows missing/unknown on key fields dropped (retains 92.4% of the 2000–2017 subset)
- `targtype1` (22 levels) and `weaptype1` (13 levels) consolidated to top-8 + "Other" to reduce one-hot sparsity and stabilize coefficients
- `attacktype1_txt`/`weaptype1_txt` have no missing values; GTD's built-in "Unknown" category is kept as valid (a pre-attack estimate can reasonably fail to resolve)
- Categorical features one-hot encoded; binary features kept as 0/1

## Method

- **Model:** Ridge Regression — chosen for interpretability (which factors associate with severity, and by how much) over raw predictive accuracy, and for coefficient stability given the one-hot dummy count
- **Loss:** Mean Squared Error
- **Split:** Chronological, not random, to test generalization to more recent unseen periods
  - Train: 2000–2014 (65.6%)
  - Validation: 2015 (13.0%)
  - Test: 2016–2017 (21.4%)

## Scope & limitations

Severity is defined narrowly as human casualties. It does not capture infrastructure damage, economic disruption, or broader societal effects — these matter for a comprehensive risk assessment but fall outside these variables.

## References

- START (National Consortium for the Study of Terrorism and Responses to Terrorism). Global Terrorism Database (GTD). University of Maryland. https://www.start.umd.edu/gtd
- Python libraries: pandas, numpy, matplotlib
