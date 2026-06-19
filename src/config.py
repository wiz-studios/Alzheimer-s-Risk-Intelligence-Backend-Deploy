from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

TARGET_COLUMN = "target"
TEST_SIZE = 0.2
RANDOM_STATE = 42
DATASET_TEST_SIZE_OVERRIDES = {
    "data1.csv": 1918,
}

MODEL_NAMES = ("random_forest", "xgboost", "knn")

DATASET_CONFIG = {
    "data1.csv": {
        "target": "target",
        "label": "Kaggle Global Alzheimer's Prediction Dataset",
        "source": "Kaggle",
        "prediction_context": "Global demographic, lifestyle, and health-factor risk classification",
    },
    "data2.csv": {
        "target": "target",
        "label": "OASIS-1 Clinical/Demographic Dataset",
        "source": "OASIS",
        "prediction_context": "Clinical/demographic dementia classification using CDR-derived target",
    },
    "data3.csv": {
        "target": "target",
        "label": "UCI DARWIN Handwriting Dataset",
        "source": "UCI Machine Learning Repository",
        "prediction_context": "Handwriting-feature Alzheimer classification",
    },
    "data4.csv": {
        "target": "target",
        "label": "CDC Healthy Aging Cognitive Decline Indicators",
        "source": "CDC Healthy Aging Data Portal",
        "prediction_context": "Population-level cognitive-decline burden classification from public-health indicators",
    },
    "data5.csv": {
        "target": "target",
        "label": "CDC Healthy Aging Mental Health Indicators",
        "source": "CDC Healthy Aging Data Portal",
        "prediction_context": "Population-level mental-health burden classification from public-health indicators",
    },
}

MODEL_INFO = {
    "random_forest": {
        "display_name": "Random Forest",
        "summary": "Tree ensemble that is robust and commonly used for tabular healthcare ML.",
    },
    "xgboost": {
        "display_name": "XGBoost",
        "summary": "Gradient-boosted tree model that often performs strongly on structured datasets.",
    },
    "knn": {
        "display_name": "KNN",
        "summary": "Distance-based baseline model; useful for comparison but sensitive to scaling.",
    },
    "hybrid_voting": {
        "display_name": "Hybrid Voting Model",
        "summary": "Soft-voting ensemble combining Random Forest and XGBoost probabilities.",
    },
}
