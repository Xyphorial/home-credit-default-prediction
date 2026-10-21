# Home Credit Default Prediction

A binary classification pipeline by Xuerun Mao. It compares logistic regression, random forest and LightGBM on the Home Credit Default Risk dataset.

## Data

Download data from https://www.kaggle.com/competitions/home-credit-default-risk/data (Kaggle access and competition rules apply). Put application_train.csv and application_test.csv in data/. For historical features, also add bureau.csv, bureau_balance.csv, previous_application.csv, installments_payments.csv, credit_card_balance.csv and POS_CASH_balance.csv. These raw files are not included.

Historical aggregation uses available files and skips absent ones. Record exactly which tables you used. The --secondary option enables history aggregation.

## Setup and execution

Use a Python environment compatible with the dependencies below. The dependency list is inferred from imports, not a tested version lock.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py --data-dir data --secondary --folds 3
```

To use only the application tables, omit --secondary. A smaller development run can use --sample 20000, but report that sampling explicitly. Running --demo generates synthetic data and must never be described as a real Home Credit result.

## Features and validation

The code handles the employment sentinel 365243, constructs financial ratios and aggregates historical numeric records by applicant using mean/max/min/count. Preprocessing fits within each training fold.

A stratified 20% holdout remains separate from selection. Development cross-validation AUC selects the model and parameters. Development out-of-fold predictions select a threshold by maximum F1. Because tuning uses the same development data, CV and OOF estimates can be optimistic. Holdout evaluation is independent of selection. After evaluation, the selected configuration is refit on all labelled data for submission predictions.

## Results

Results supplied from a real HOME_CREDIT run: 307,511 labelled rows, 126 input features before encoding, positive fraction 8.0729%, Python 3.12.14, seed 42. Historical-table usage and the exact run command remain unconfirmed.

| Model | Development CV AUC | Holdout AUC | Holdout AP |
|---|---:|---:|---:|
| LightGBM | 0.760748 | 0.764897 | 0.256499 |
| Logistic regression | 0.749182 | 0.752069 | 0.237755 |
| Random forest | 0.740086 | 0.744314 | 0.231170 |

Development CV selected LightGBM (num_leaves=15, reg_lambda=5). The development OOF F1 threshold was 0.156324. Holdout precision: 0.256679, recall: 0.414099, F1: 0.316917. Holdout accuracy was not included in the supplied metrics. Results do not establish profit improvement or statistical significance of model differences. Logistic C=0.1; random forest max_depth=10.

Outputs include ROC and precision–recall curves, a confusion matrix, model comparison and interpretation plots. Kaggle application_test has no labels, so the script does not report a local test AUC. Permutation importance explains the selected model on development data. SHAP explains LightGBM specifically.

## Limitations

Historical inputs need a field-level audit of availability at application time. The implementation does not explicitly filter all historical fields by time. F1 optimization is not a profit-maximizing lending policy. Probability calibration, loss assumptions and customer-specific economics require further work. Importance does not establish causality.

## Reproducibility

Random seed: 42. Default CV folds: 3. Record the exact command, Python version, installed package versions and source data used for your final run. Never publish applicant-level prediction files or raw data with the presentation.
