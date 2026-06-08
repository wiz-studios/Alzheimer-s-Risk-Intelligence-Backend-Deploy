# Real Dataset Notes

These CSVs are prepared from public datasets for Part B practical development.
Version 5 applies the row-count feedback by sampling the Kaggle source so the prepared total is exactly 10,000 rows.

| File | Source | Rows | Target definition | Limitation |
| --- | --- | ---: | --- | --- |
| data1.csv | Kaggle Alzheimer's Prediction Dataset (Global), stratified sample | 9591 | `Alzheimer's Diagnosis = Yes` | Public Kaggle dataset sampled from the larger source to satisfy the approximate 10,000-row coursework requirement. |
| data2.csv | OASIS-1 cross-sectional demographic and clinical data | 235 | `CDR > 0` | Small subject-level MRI/clinical cohort; predicts dementia classification, not medical diagnosis. |
| data3.csv | UCI DARWIN handwriting Alzheimer dataset | 174 | `P = Alzheimer patient`, `H = healthy control` | Small subject-level handwriting dataset; high-dimensional features. |

Prepared total rows: 10000.
Do not describe these outputs as clinical diagnosis. The system should be described as an academic risk/classification prototype.