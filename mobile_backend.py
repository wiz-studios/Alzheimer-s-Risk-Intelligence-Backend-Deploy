import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import joblib
import pandas as pd

from src.config import DATA_RAW_DIR, DATASET_CONFIG, MODEL_INFO, MODELS_DIR, RESULTS_DIR
from src.data_loader import get_target_column, list_available_datasets
from src.inspection import (
    extract_feature_importance,
    local_interpretation_text,
    model_family,
    pipeline_summary,
    shap_available,
    shap_explanation,
)
from src.predict import predict_risk, risk_explanation


HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "8510"))
MODEL_CACHE = {}


def load_json(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataset(dataset_name):
    if dataset_name not in list_available_datasets():
        raise ValueError(f"Unknown dataset: {dataset_name}")
    df = pd.read_csv(DATA_RAW_DIR / dataset_name)
    target = get_target_column(dataset_name)
    if target not in df.columns:
        raise ValueError(f"Target column missing: {target}")
    return df, target


def model_key_from_filename(dataset_name, model_filename):
    prefix = f"{dataset_name.replace('.csv', '')}_"
    return model_filename.replace(prefix, "").replace(".joblib", "")


def model_filename(dataset_name, model_key):
    return f"{dataset_name.replace('.csv', '')}_{model_key}.joblib"


def load_model(dataset_name, model_key):
    cache_key = f"{dataset_name}:{model_key}"
    if cache_key not in MODEL_CACHE:
        path = MODELS_DIR / model_filename(dataset_name, model_key)
        if not path.exists():
            raise ValueError(f"Model not found: {path.name}")
        MODEL_CACHE[cache_key] = joblib.load(path)
    return MODEL_CACHE[cache_key]


def available_models(dataset_name):
    prefix = f"{dataset_name.replace('.csv', '')}_"
    models = []
    for path in sorted(MODELS_DIR.glob(f"{prefix}*.joblib")):
        key = model_key_from_filename(dataset_name, path.name)
        family = model_family(key)
        models.append(
            {
                "key": key,
                "filename": path.name,
                "display_name": MODEL_INFO.get(family, {}).get("display_name", key),
                "summary": MODEL_INFO.get(family, {}).get("summary", ""),
            }
        )
    return models


def recommended_model(metrics, mode="overall"):
    valid = {name: values for name, values in metrics.items() if isinstance(values, dict)}
    if not valid:
        return None
    if mode == "sensitivity":
        return max(valid, key=lambda name: valid[name].get("recall", 0))
    if mode == "precision":
        return max(valid, key=lambda name: valid[name].get("precision", 0))
    if mode == "calibration":
        return min(valid, key=lambda name: valid[name].get("brier_score", 1))
    if mode == "balanced":
        return max(
            valid,
            key=lambda name: (
                valid[name].get("roc_auc", 0)
                + valid[name].get("f1_score", 0)
                + valid[name].get("recall", 0)
                + valid[name].get("precision", 0)
            )
            / 4,
        )
    return max(valid, key=lambda name: valid[name].get("roc_auc", 0))


def validate_input(input_data, feature_template):
    row = pd.DataFrame([input_data])
    missing = [column for column in feature_template.columns if column not in row.columns]
    if missing:
        raise ValueError(f"Missing input fields: {', '.join(missing[:10])}")

    row = row[feature_template.columns].copy()
    for column, dtype in feature_template.dtypes.items():
        if pd.api.types.is_numeric_dtype(dtype):
            row[column] = pd.to_numeric(row[column], errors="raise")
        else:
            row[column] = row[column].astype(object).where(row[column].notna(), None)

    if "Age" in row.columns:
        age = float(row["Age"].iloc[0])
        if age < 0 or age > 120:
            raise ValueError("Age must be between 0 and 120.")
    if "BMI" in row.columns:
        bmi = float(row["BMI"].iloc[0])
        if bmi < 10 or bmi > 80:
            raise ValueError("BMI must be between 10 and 80.")
    if "Cognitive Test Score" in row.columns:
        score = float(row["Cognitive Test Score"].iloc[0])
        if score < 0 or score > 100:
            raise ValueError("Cognitive Test Score must be between 0 and 100.")
    return row


def compact_metrics(metrics):
    output = {}
    keys = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "average_precision",
        "brier_score",
        "confusion_matrix",
        "subgroup_metrics",
        "calibration_curve",
        "roc_curve",
        "precision_recall_curve",
    ]
    for model_key, values in metrics.items():
        if isinstance(values, dict):
            output[model_key] = {key: values.get(key) for key in keys if key in values}
    return output


def metadata_summary(metadata):
    output = {}
    for model_key, values in metadata.items():
        if isinstance(values, dict):
            output[model_key] = {
                "dataset_label": values.get("dataset_label"),
                "model_class": values.get("model_class"),
                "train_rows": values.get("train_rows"),
                "test_rows": values.get("test_rows"),
                "feature_count": values.get("feature_count"),
                "random_state": values.get("random_state"),
                "target_column": values.get("target_column"),
                "saved_as_pipeline": values.get("saved_as_pipeline"),
                "pipeline_steps": values.get("pipeline_steps"),
                "preprocessing": values.get("preprocessing"),
                "trained_at_utc": values.get("trained_at_utc"),
                "test_fraction": values.get("test_fraction"),
            }
    return output


class MobileBackendHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)

            if parsed.path == "/api/health":
                self._json(
                    {
                        "status": "ready",
                        "service": "Alzheimer's Risk Intelligence backend",
                        "version": "native-mobile-v1",
                    }
                )
                return

            if parsed.path == "/":
                self._json(
                    {
                        "service": "Alzheimer's Risk Intelligence backend",
                        "status": "ready",
                        "api_base": "/api",
                        "health": "/api/health",
                    }
                )
                return

            if parsed.path == "/api/datasets":
                datasets = []
                for dataset in list_available_datasets():
                    df, target = load_dataset(dataset)
                    config = DATASET_CONFIG.get(dataset, {})
                    datasets.append(
                        {
                            "name": dataset,
                            "label": config.get("label", dataset),
                            "source": config.get("source", "Unknown"),
                            "context": config.get("prediction_context", ""),
                            "rows": len(df),
                            "features": len(df.columns) - 1,
                            "target_column": target,
                            "positive_rate": round(float(df[target].mean()), 4),
                            "columns": [column for column in df.columns if column != target],
                        }
                    )
                self._json({"datasets": datasets})
                return

            if parsed.path == "/api/models":
                dataset = query.get("dataset", ["data1.csv"])[0]
                mode = query.get("mode", ["overall"])[0]
                metrics = load_json(RESULTS_DIR / f"{dataset.replace('.csv', '')}_metrics.json")
                self._json(
                    {
                        "dataset": dataset,
                        "recommendation_mode": mode,
                        "recommended": recommended_model(metrics, mode),
                        "models": available_models(dataset),
                    }
                )
                return

            if parsed.path == "/api/warmup":
                dataset = query.get("dataset", ["data1.csv"])[0]
                model_key = query.get("model", ["xgboost"])[0]
                model = load_model(dataset, model_key)
                self._json(
                    {
                        "dataset": dataset,
                        "model": model_key,
                        "status": "loaded",
                        "pipeline": pipeline_summary(model),
                    }
                )
                return

            if parsed.path == "/api/sample":
                dataset = query.get("dataset", ["data1.csv"])[0]
                row_index = int(query.get("row", [0])[0])
                df, target = load_dataset(dataset)
                features = df.drop(columns=[target])
                if row_index not in features.index:
                    row_index = int(features.index[0])
                row = features.loc[row_index].where(pd.notna(features.loc[row_index]), None).to_dict()
                self._json({"dataset": dataset, "row_index": row_index, "input": row})
                return

            if parsed.path == "/api/metrics":
                dataset = query.get("dataset", ["data1.csv"])[0]
                metrics = load_json(RESULTS_DIR / f"{dataset.replace('.csv', '')}_metrics.json")
                self._json({"dataset": dataset, "metrics": compact_metrics(metrics)})
                return

            if parsed.path == "/api/metadata":
                dataset = query.get("dataset", ["data1.csv"])[0]
                metadata = load_json(RESULTS_DIR / f"{dataset.replace('.csv', '')}_model_metadata.json")
                self._json({"dataset": dataset, "metadata": metadata_summary(metadata)})
                return

            if parsed.path == "/api/source-notes":
                path = DATA_RAW_DIR.parent / "source_notes.md"
                text = path.read_text(encoding="utf-8") if path.exists() else "Source notes unavailable."
                self._json({"source_notes": text})
                return

            self._json({"error": "Not found"}, status=404)
        except Exception as exc:
            self._json({"error": str(exc)}, status=400)

    def do_POST(self):
        try:
            parsed = urlparse(self.path)
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            payload = json.loads(body or "{}")

            if parsed.path != "/api/predict":
                self._json({"error": "Not found"}, status=404)
                return

            dataset = payload.get("dataset", "data1.csv")
            model_key = payload.get("model", "xgboost")
            include_shap = bool(payload.get("include_shap", True))
            input_data = payload.get("input", {})

            df, target = load_dataset(dataset)
            features = df.drop(columns=[target])
            input_row = validate_input(input_data, features)

            model = load_model(dataset, model_key)
            prediction = predict_risk(model, input_row)
            importance = extract_feature_importance(model)
            shap_df = pd.DataFrame()
            shap_message = "SHAP not requested."
            if include_shap and shap_available():
                shap_df, shap_message = shap_explanation(model, features, input_row)
            elif include_shap:
                shap_message = "SHAP is not installed in this backend environment."

            self._json(
                {
                    "dataset": dataset,
                    "model": model_key,
                    "model_display_name": MODEL_INFO.get(model_family(model_key), {}).get("display_name", model_key),
                    "prediction": prediction.iloc[0].to_dict(),
                    "risk_explanation": risk_explanation(float(prediction.loc[0, "risk_probability"])),
                    "local_interpretation": local_interpretation_text(input_row, importance),
                    "pipeline": pipeline_summary(model),
                    "top_features": importance.head(12).to_dict(orient="records"),
                    "shap_message": shap_message,
                    "shap_values": shap_df.head(10).to_dict(orient="records") if not shap_df.empty else [],
                    "disclaimer": "Academic prototype only. Not a clinical diagnostic tool.",
                }
            )
        except Exception as exc:
            self._json({"error": str(exc)}, status=400)

    def _json(self, payload, status=200):
        self.send_response(status)
        self._cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(payload, default=str).encode("utf-8"))

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, format, *args):
        return


def main():
    server = ThreadingHTTPServer((HOST, PORT), MobileBackendHandler)
    print(f"Alzheimer's Risk Intelligence backend running at http://localhost:{PORT}")
    print("For a physical phone, use this computer's LAN IP instead of localhost.")
    server.serve_forever()


if __name__ == "__main__":
    main()
