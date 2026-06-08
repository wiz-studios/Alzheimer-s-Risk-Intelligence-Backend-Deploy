import argparse
import json
from datetime import datetime, timezone

import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from .config import DATASET_CONFIG, DATASET_TEST_SIZE_OVERRIDES, MODELS_DIR, RANDOM_STATE, RESULTS_DIR, TEST_SIZE
from .data_loader import get_target_column, load_dataset
from .evaluate import evaluate_classifier, save_metrics
from .preprocess import build_preprocessor, make_train_test_split


def build_models() -> dict:
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=14,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "knn": KNeighborsClassifier(n_neighbors=7),
    }

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        )
    except ImportError:
        print("xgboost is not installed. Skipping XGBoost model.")

    if "random_forest" in models and "xgboost" in models:
        models["hybrid_voting"] = VotingClassifier(
            estimators=[
                ("random_forest", models["random_forest"]),
                ("xgboost", models["xgboost"]),
            ],
            voting="soft",
        )

    return models


def train_and_evaluate(dataset_name: str) -> dict:
    df = load_dataset(dataset_name)
    target_column = get_target_column(dataset_name)
    split_test_size = DATASET_TEST_SIZE_OVERRIDES.get(dataset_name, TEST_SIZE)
    X_train, X_test, y_train, y_test = make_train_test_split(df, target_column, split_test_size)
    preprocessor = build_preprocessor(X_train)

    metrics_by_model = {}
    metadata_by_model = {}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, classifier in build_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )
        pipeline.fit(X_train, y_train)

        metrics = evaluate_classifier(pipeline, X_test, y_test)
        metrics_by_model[model_name] = metrics

        model_path = MODELS_DIR / f"{dataset_name.replace('.csv', '')}_{model_name}.joblib"
        joblib.dump(pipeline, model_path)
        print(f"Saved model: {model_path}")
        metadata_by_model[model_name] = build_model_metadata(
            dataset_name=dataset_name,
            model_name=model_name,
            pipeline=pipeline,
            X_train=X_train,
            X_test=X_test,
            target_column=target_column,
            split_test_size=split_test_size,
        )

    metrics_path = RESULTS_DIR / f"{dataset_name.replace('.csv', '')}_metrics.json"
    save_metrics(metrics_by_model, metrics_path)
    print(f"Saved metrics: {metrics_path}")

    metadata_path = RESULTS_DIR / f"{dataset_name.replace('.csv', '')}_model_metadata.json"
    metadata_path.write_text(json.dumps(metadata_by_model, indent=2, default=str), encoding="utf-8")
    print(f"Saved metadata: {metadata_path}")

    return metrics_by_model


def build_model_metadata(
    dataset_name: str,
    model_name: str,
    pipeline,
    X_train,
    X_test,
    target_column: str,
    split_test_size,
) -> dict:
    classifier = pipeline.named_steps.get("classifier")
    return {
        "dataset": dataset_name,
        "dataset_label": DATASET_CONFIG.get(dataset_name, {}).get("label", dataset_name),
        "model": model_name,
        "model_class": type(classifier).__name__ if classifier is not None else "Unknown",
        "target_column": target_column,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_count": int(len(X_train.columns)),
        "feature_columns": list(X_train.columns),
        "holdout_indices": [int(index) for index in X_test.index.tolist()],
        "test_size": split_test_size,
        "test_fraction": float(len(X_test) / (len(X_train) + len(X_test))),
        "random_state": RANDOM_STATE,
        "saved_as_pipeline": True,
        "pipeline_steps": list(pipeline.named_steps.keys()),
        "preprocessing": [
            "median imputation and standard scaling for numeric features",
            "most-frequent imputation and one-hot encoding for categorical features",
        ],
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "classifier_params": classifier.get_params() if hasattr(classifier, "get_params") else {},
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train Alzheimer risk prediction models.")
    parser.add_argument("--dataset", required=True, help="CSV file name under data/raw, e.g. data1.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_evaluate(args.dataset)
