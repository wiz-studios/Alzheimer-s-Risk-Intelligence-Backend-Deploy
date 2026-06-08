import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.calibration import calibration_curve


def evaluate_classifier(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        y_score = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        y_score = y_pred

    fpr, tpr, _ = roc_curve(y_test, y_score)
    pr_precision, pr_recall, _ = precision_recall_curve(y_test, y_score)
    prob_true, prob_pred = calibration_curve(y_test, y_score, n_bins=10, strategy="quantile")

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "average_precision": float(average_precision_score(y_test, y_score)),
        "brier_score": float(brier_score_loss(y_test, y_score)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "roc_curve": {
            "fpr": [float(value) for value in fpr],
            "tpr": [float(value) for value in tpr],
        },
        "precision_recall_curve": {
            "precision": [float(value) for value in pr_precision],
            "recall": [float(value) for value in pr_recall],
        },
        "calibration_curve": {
            "predicted_probability": [float(value) for value in prob_pred],
            "observed_frequency": [float(value) for value in prob_true],
        },
        "subgroup_metrics": subgroup_metrics(X_test, y_test, y_pred),
    }
    return metrics


def subgroup_metrics(X_test: pd.DataFrame, y_test, y_pred) -> dict:
    output = {}
    candidates = [
        "Gender",
        "sex",
        "M/F",
        "Age",
        "Country",
        "Employment Status",
        "Urban vs Rural Living",
    ]

    result_frame = X_test.copy()
    result_frame["_actual"] = list(y_test)
    result_frame["_predicted"] = list(y_pred)

    for column in candidates:
        if column not in result_frame.columns:
            continue

        if column == "Age":
            groups = pd.cut(
                pd.to_numeric(result_frame[column], errors="coerce"),
                bins=[0, 59, 69, 79, 120],
                labels=["under_60", "60_69", "70_79", "80_plus"],
                include_lowest=True,
            )
        else:
            unique_count = result_frame[column].nunique(dropna=True)
            if unique_count > 12:
                continue
            groups = result_frame[column].astype("string").fillna("missing")

        grouped = result_frame.assign(_group=groups).dropna(subset=["_group"]).groupby("_group")
        rows = []
        for group_name, group in grouped:
            if len(group) < 5:
                continue
            rows.append(
                {
                    "group": str(group_name),
                    "count": int(len(group)),
                    "positive_rate": float(group["_actual"].mean()),
                    "accuracy": float(accuracy_score(group["_actual"], group["_predicted"])),
                    "precision": float(precision_score(group["_actual"], group["_predicted"], zero_division=0)),
                    "recall": float(recall_score(group["_actual"], group["_predicted"], zero_division=0)),
                    "f1_score": float(f1_score(group["_actual"], group["_predicted"], zero_division=0)),
                }
            )

        if rows:
            output[column] = rows

    return output


def save_metrics(metrics_by_model: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics_by_model, indent=2), encoding="utf-8")

    rows = []
    for model_name, metrics in metrics_by_model.items():
        row = {"model": model_name}
        row.update(
            {
                key: value
                for key, value in metrics.items()
                if key
                not in {
                    "confusion_matrix",
                    "roc_curve",
                    "precision_recall_curve",
                    "calibration_curve",
                    "subgroup_metrics",
                }
            }
        )
        rows.append(row)

    csv_path = output_path.with_suffix(".csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
