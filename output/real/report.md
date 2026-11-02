# Home Credit results

Real application CSV run.

| model               |   selection_cv_auc |   fold_auc_std |   oof_auc | parameters                                        |   holdout_auc |   holdout_average_precision |
|:--------------------|-------------------:|---------------:|----------:|:--------------------------------------------------|--------------:|----------------------------:|
| LightGBM            |           0.760748 |    2.28079e-05 |  0.760728 | {"model__num_leaves": 15, "model__reg_lambda": 5} |      0.764897 |                    0.256499 |
| Logistic regression |           0.749182 |    0.00157888  |  0.749182 | {"model__C": 0.1}                                 |      0.752069 |                    0.237755 |
| Random forest       |           0.740086 |    0.0016043   |  0.74007  | {"model__max_depth": 10}                          |      0.744314 |                    0.23117  |

```json
{
  "data_mode": "HOME_CREDIT",
  "selected_model": "LightGBM",
  "selection_rule": "highest development CV AUC, before holdout evaluation",
  "threshold_from_development_oof": 0.15632379569035873,
  "holdout_auc": 0.7648971610214157,
  "holdout_average_precision": 0.25649892779957334,
  "precision": 0.25667915106117356,
  "recall": 0.41409869083585094,
  "f1": 0.31691714836223506,
  "default_fraction": 0.08072881945686496,
  "rows": 307511,
  "features": 126,
  "python": "3.12.14",
  "seed": 42,
  "cv_note": "CV and OOF are used for tuning; final holdout is the independent evaluation. Threshold OOF is not nested."
}
```

SHAP explains the LightGBM model; permutation importance explains the selected model. Neither establishes causation. No labels exist for Kaggle application_test, so no test AUC is reported.
