from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def load_model(model_path: str | Path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


def predict_risk(model, input_data: pd.DataFrame) -> pd.DataFrame:
    probabilities = estimate_positive_probability(model, input_data)
    predictions = model.predict(input_data)
    uncertainty = estimate_uncertainty(model, input_data, probabilities)

    return pd.DataFrame(
        {
            "risk_probability": probabilities,
            "risk_percent": (probabilities * 100).round(2),
            "predicted_class": predictions,
            "risk_level": [risk_level(score) for score in probabilities],
            "uncertainty_level": [item["level"] for item in uncertainty],
            "uncertainty_note": [item["note"] for item in uncertainty],
        }
    )


def risk_level(probability: float) -> str:
    if probability < 0.33:
        return "Low model-estimated probability"
    if probability < 0.66:
        return "Moderate model-estimated probability"
    return "Elevated model-estimated probability"


def risk_explanation(probability: float) -> str:
    level = risk_level(probability)
    if level.startswith("Low"):
        return "Low risk means the model found fewer risk patterns in this sample."
    if level.startswith("Moderate"):
        return "Moderate probability means the model found some risk patterns and the case should be reviewed carefully."
    return "Elevated probability means the model found stronger risk patterns. This still requires professional medical review."


def estimate_positive_probability(model, input_data: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(input_data)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(input_data)
        scores = np.asarray(scores, dtype=float)
        score_range = scores.max() - scores.min()
        if score_range == 0:
            return np.full(scores.shape, 0.5, dtype=float)
        return (scores - scores.min()) / score_range
    return model.predict(input_data).astype(float)


def estimate_uncertainty(model, input_data: pd.DataFrame, probabilities: np.ndarray) -> list[dict]:
    estimator_probs = _voting_estimator_probabilities(model, input_data)
    output = []

    for index, probability in enumerate(probabilities):
        distance_from_boundary = abs(float(probability) - 0.5)
        disagreement = None
        if estimator_probs is not None:
            disagreement = float(np.std(estimator_probs[:, index]))

        if disagreement is not None and disagreement >= 0.18:
            level = "High"
            note = "High uncertainty: ensemble models disagree noticeably."
        elif distance_from_boundary < 0.1:
            level = "High"
            note = "High uncertainty: probability is close to the decision boundary."
        elif distance_from_boundary < 0.2 or (disagreement is not None and disagreement >= 0.1):
            level = "Moderate"
            note = "Moderate uncertainty: interpret this estimate carefully."
        else:
            level = "Lower"
            note = "Lower uncertainty: probability is farther from the decision boundary."

        output.append({"level": level, "note": note})

    return output


def _voting_estimator_probabilities(model, input_data: pd.DataFrame):
    if not hasattr(model, "named_steps"):
        return None
    preprocessor = model.named_steps.get("preprocessor")
    classifier = model.named_steps.get("classifier")
    if preprocessor is None or classifier is None or not hasattr(classifier, "estimators_"):
        return None

    transformed = preprocessor.transform(input_data)
    probs = []
    for estimator in classifier.estimators_:
        if hasattr(estimator, "predict_proba"):
            probs.append(estimator.predict_proba(transformed)[:, 1])
    if not probs:
        return None
    return np.vstack(probs)
