from pathlib import Path

import pandas as pd

from .config import DATA_RAW_DIR, DATASET_CONFIG, TARGET_COLUMN


def list_available_datasets() -> list[str]:
    return sorted(path.name for path in DATA_RAW_DIR.glob("*.csv"))


def load_dataset(dataset_name: str) -> pd.DataFrame:
    dataset_path = Path(dataset_name)
    if not dataset_path.is_absolute():
        dataset_path = DATA_RAW_DIR / dataset_name

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    target_column = DATASET_CONFIG.get(dataset_name, {}).get("target", TARGET_COLUMN)
    if target_column not in df.columns:
        raise ValueError(
            f"Dataset must include target column '{target_column}'. "
            f"Found columns: {', '.join(df.columns)}"
        )

    return df


def get_target_column(dataset_name: str) -> str:
    return DATASET_CONFIG.get(dataset_name, {}).get("target", TARGET_COLUMN)
