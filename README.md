# Alzheimer's Risk Intelligence Backend

This backend exposes the v5 machine-learning assets to the native Android app.

It keeps the same practical system assets:

- `data/raw/data1.csv`, `data2.csv`, `data3.csv`, `data4.csv`, `data5.csv`
- trained sklearn/joblib pipelines in `models/`
- generated metrics and metadata in `results/`
- prediction, uncertainty, feature importance, and SHAP support through `src/`

## Run

```powershell
run_backend.bat
```

or:

```powershell
python mobile_backend.py
```

The API runs on:

```text
http://localhost:8510/api
```

For a physical phone, use the computer's Wi-Fi/LAN IP address in the Android app, for example:

```text
http://192.168.1.20:8510/api
```

## Main Endpoints

- `GET /api/health`
- `GET /api/datasets`
- `GET /api/models?dataset=data1.csv&mode=overall`
- `GET /api/warmup?dataset=data1.csv&model=xgboost`
- `GET /api/sample?dataset=data1.csv`
- `GET /api/metrics?dataset=data1.csv`
- `GET /api/metadata?dataset=data1.csv`
- `GET /api/source-notes`
- `POST /api/predict`

This is an academic decision-support backend, not a clinical diagnostic service.
