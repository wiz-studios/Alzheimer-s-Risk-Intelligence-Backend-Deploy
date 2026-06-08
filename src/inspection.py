from __future__ import annotations

import numpy as np
import pandas as pd


def model_family(model_key: str) -> str:
    if model_key.startswith("random_forest"):
        return "random_forest"
    if model_key.startswith("xgboost"):
        return "xgboost"
    if model_key.startswith("knn"):
        return "knn"
    if model_key.startswith("hybrid"):
        return "hybrid_voting"
    return model_key


def extract_feature_importance(pipeline, top_n: int = 12) -> pd.DataFrame:
    if not hasattr(pipeline, "named_steps"):
        return pd.DataFrame()

    preprocessor = pipeline.named_steps.get("preprocessor")
    classifier = pipeline.named_steps.get("classifier")
    if preprocessor is None or classifier is None:
        return pd.DataFrame()

    if not hasattr(classifier, "feature_importances_"):
        return pd.DataFrame()

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        return pd.DataFrame()

    values = classifier.feature_importances_
    if len(feature_names) != len(values):
        return pd.DataFrame()

    importance = pd.DataFrame(
        {
            "feature": [name.replace("numeric__", "").replace("categorical__", "") for name in feature_names],
            "importance": values,
        }
    )
    return importance.sort_values("importance", ascending=False).head(top_n)


def pipeline_summary(pipeline) -> dict:
    if not hasattr(pipeline, "named_steps"):
        return {"type": type(pipeline).__name__}

    classifier = pipeline.named_steps.get("classifier")
    return {
        "pipeline_type": type(pipeline).__name__,
        "steps": list(pipeline.named_steps.keys()),
        "classifier": type(classifier).__name__ if classifier is not None else "Unknown",
    }


def local_interpretation_text(input_row: pd.DataFrame, importance: pd.DataFrame, max_items: int = 4) -> str:
    if importance.empty:
        return "Local explanation is unavailable for this model. Review the global model metrics and source limitations."

    messages = []
    for feature in importance["feature"].head(max_items):
        raw_feature = feature.split("_", 1)[0] if "_" in feature else feature
        if raw_feature in input_row.columns:
            messages.append(f"{raw_feature} = {input_row[raw_feature].iloc[0]}")

    if not messages:
        return (
            "The prediction is most influenced by encoded model features shown in the feature importance chart. "
            "For categorical data, one-hot encoded feature names may not map directly to a single visible input field."
        )

    return (
        "The model's strongest global drivers for this type of prediction include "
        + ", ".join(messages)
        + ". Interpret this as a model pattern explanation, not a clinical cause."
    )


def shap_available() -> bool:
    try:
        import shap  # noqa: F401

        return True
    except Exception:
        return False


def shap_explanation(pipeline, background: pd.DataFrame, input_row: pd.DataFrame, top_n: int = 10) -> tuple[pd.DataFrame, str]:
    """Return approximate local SHAP contributions for tree classifiers when SHAP is installed."""
    try:
        import shap
    except Exception:
        return pd.DataFrame(), "SHAP is not installed. Install requirements.txt to enable SHAP explanations."

    if not hasattr(pipeline, "named_steps"):
        return pd.DataFrame(), "SHAP requires a saved sklearn Pipeline."

    preprocessor = pipeline.named_steps.get("preprocessor")
    classifier = pipeline.named_steps.get("classifier")
    if preprocessor is None or classifier is None:
        return pd.DataFrame(), "SHAP requires a pipeline with preprocessor and classifier steps."

    supported = {"RandomForestClassifier", "XGBClassifier"}
    if type(classifier).__name__ not in supported:
        return pd.DataFrame(), "SHAP panel is available for Random Forest and XGBoost models in this prototype."

    try:
        sample_background = background.head(min(len(background), 200))
        transformed_background = preprocessor.transform(sample_background)
        transformed_input = preprocessor.transform(input_row)
        feature_names = preprocessor.get_feature_names_out()

        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(transformed_input)
        values = _positive_class_shap_values(shap_values)

        result = pd.DataFrame(
            {
                "feature": [name.replace("numeric__", "").replace("categorical__", "") for name in feature_names],
                "shap_value": values,
                "absolute_impact": np.abs(values),
            }
        )
        result = result.sort_values("absolute_impact", ascending=False).head(top_n)
        return result, "SHAP values estimate local feature contribution for this specific prediction."
    except Exception as exc:
        return pd.DataFrame(), f"SHAP explanation failed: {exc}"


def _positive_class_shap_values(shap_values) -> np.ndarray:
    if isinstance(shap_values, list):
        selected = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        selected = shap_values

    selected = np.asarray(selected)
    if selected.ndim == 3:
        selected = selected[:, :, 1]
    if selected.ndim == 2:
        selected = selected[0]
    return selected.astype(float)
