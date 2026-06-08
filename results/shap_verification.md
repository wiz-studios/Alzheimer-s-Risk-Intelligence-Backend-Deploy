# SHAP Verification

SHAP was installed and verified in the v5 execution environment after retraining the corrected 10,000-row dataset.

Verified command context:

```text
python package: shap
version: 0.51.0
dataset: data1.csv, 9,591-row sampled Kaggle dataset
models checked: data1_xgboost.joblib, data1_random_forest.joblib
explanation type: local SHAP explanation for one selected row
status: passed
```

Observed output summary:

| Model | SHAP rows returned | Status |
| --- | ---: | --- |
| data1_xgboost.joblib | 5 | passed |
| data1_random_forest.joblib | 5 | passed |

Important limitation:

SHAP explains model contribution behavior for a prediction. It does not establish clinical causation, medical validity, or diagnostic reliability.
