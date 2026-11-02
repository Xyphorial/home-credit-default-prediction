# Home Credit Default Risk | End-to-End Credit Risk Modelling

**Python • LightGBM • Feature Engineering • Stratified CV • SHAP**

Estimate the probability of a Home Credit applicant experiencing repayment difficulties. This is an analytical ML classification project using the [Kaggle Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) dataset. Raw competition data and row-level customer predictions are **not redistributed**.

## Overview

- Compared Logistic Regression, Random Forest, primary-application LightGBM, bureau-enhanced LightGBM and a **full-history LightGBM**.
- Generated application ratios, EXT_SOURCE summary features, and six historical-table aggregates joined at applicant level.
- Split the available labelled dataset chronologically *not* by date but using a fixed **stratified random 80/20 holdout**, with threefold stratified CV inside development data.
- Selected hyperparameters by CV AUC and operating threshold using **development-only out-of-fold** predictions; evaluated on a separate holdout.
- Investigated feature importance, SHAP explanations, missing data, probability ranking and error subgroups.

## Archived results

| Model | Holdout ROC AUC |
|---|---:|
| Logistic Regression | 0.7521 |
| Random Forest | 0.7443 |
| Application-only LightGBM | 0.7649 |
| LightGBM + bureau | 0.7691 |
| **LightGBM + all available history** | **0.7822** |

Best full-history run: **470 features**, **OOF ROC AUC 0.7792**, **holdout average precision 0.2838**, 307,511 labelled training rows and 48,744 unlabelled competition test rows. There is **no labelled Kaggle-test AUC** or reported live competition score; 0.7822 is an internal holdout result.

See [full results (Chinese)](RESULTS_ZH.md), [model comparison](output/all_model_comparison.csv), [metrics](output/full_history/metrics.json), [ROC plot](output/full_history/roc_curve.png), [PR plot](output/full_history/pr_curve.png) and [feature importance](output/full_history/feature_importance.png).

## Running the project

1. Install Python 3.12 and the dependencies in `requirements.txt` (original tested versions in `requirements-tested.txt`).
2. Download the competition files from Kaggle into `data/`; [see required filenames](data/README.md).
3. Run stages in this order: `main.py --folds 3`, `enhanced.py`, `enhanced.py --extra`, `finalize_report.py`, `verify_outputs.py`. `run_project.py` is an orchestration helper.
4. Open `walkthrough.ipynb` to examine the **included aggregate results and plots** without retraining.

The complete raw-data-based verification, row-level prediction tables and final model binaries are intentionally omitted from the public version. They can be recreated locally after downloading data and running the pipeline. Scripts that check full saved predictions require those regenerated outputs.

## File guide

- `main.py` — baseline EDA, model training, CV, evaluation and outputs.
- `history.py` and `extra_history.py` — applicant-level historical aggregations.
- `enhanced.py` — enriched models and selection.
- `finalize_report.py` and `verify_outputs.py` — reporting and audits.
- `walkthrough.ipynb` — visually explores included results.
- `output/` — selected summary CSVs, JSON metrics and figures (not individual customer records).

## Caveats

This is an observational ranking model, not a causal model or production underwriting system. Internal random-holdout performance does not establish future-calendar stability. Model searches were limited, probabilities were not calibrated, and optimizing F1 is not equivalent to optimizing a lending decision's economic cost. Feature generation must account for data availability at application time. Data/label meanings and feature caveats are discussed in `FEATURE_GUIDE_ZH.md` and `DATE_AUDIT.md`.
