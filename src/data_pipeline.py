"""
data_pipeline.py
-----------------
Single source of truth for:
  - which columns exist and what they mean
  - valid input ranges (used for data validation in the API and UI)
  - how raw patient data is turned into model-ready features

Every other module (train.py, api/main.py, app/streamlit_app.py) imports
from here instead of redefining preprocessing logic. This is what makes
the pipeline consistent between training and prediction time.
"""

from dataclasses import dataclass
from pathlib import Path
import hashlib
from typing import Dict, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data/heart.csv"
DATA_HASH = "e31e52eb5ee890c1a11a3baef3df95e8174718d5b94b0a8b77665ec3c07328ff"
TARGET_COLUMN = "target"

# Columns that are genuinely continuous / ordinal-numeric and get scaled.
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]

# Columns that are categorical codes even though they're stored as integers
# (e.g. cp=0..3 is a chest-pain *category*, not a magnitude) and get one-hot
# encoded rather than scaled.
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Human-readable labels, used in the UI, PDF report and SHAP plots so a
# non-technical reader sees "Chest Pain Type" instead of "cp".
FEATURE_LABELS: Dict[str, str] = {
    "age": "Age",
    "trestbps": "Resting Blood Pressure",
    "chol": "Cholesterol",
    "thalach": "Max Heart Rate Achieved",
    "oldpeak": "ST Depression (Oldpeak)",
    "ca": "Major Vessels Colored by Fluoroscopy",
    "sex": "Sex",
    "cp": "Chest Pain Type",
    "fbs": "Fasting Blood Sugar > 120 mg/dl",
    "restecg": "Resting ECG Result",
    "exang": "Exercise-Induced Angina",
    "slope": "Slope of Peak Exercise ST Segment",
    "thal": "Thallium stress-test result",
}

# (min, max, step) — used for both Streamlit number_input widgets and
# server-side validation in the API. Ranges are set a little wider than the
# observed data range so genuine (if unusual) patients aren't rejected, but
# tight enough to catch obvious data-entry errors (age = -5, chol = 900...).
NUMERIC_RANGES: Dict[str, Tuple[float, float, float]] = {
    "age": (1, 120, 1),
    "trestbps": (60, 220, 1),      # resting BP in mmHg (training data: 94-200)
    "chol": (50, 605, 1),          # serum cholesterol mg/dl (training data: 126-564 — tree
                                    # models can't extrapolate past the highest value they
                                    # ever split on, so predictions go flat well beyond this)
    "thalach": (50, 250, 1),       # max heart rate achieved (training data: 71-202)
    "oldpeak": (0.0, 6.5, 0.1),    # ST depression (training data: 0.0-6.2)
    "ca": (0, 3, 1),               # number of major vessels colored (4 was a missing-data code — see load_dataset)
}

# Category code -> human label, used for Streamlit dropdowns.
CATEGORY_OPTIONS: Dict[str, Dict[int, str]] = {
    "sex": {1: "Male", 0: "Female"},
    "cp": {
        0: "Asymptomatic",
        1: "Atypical angina",
        2: "Non-anginal pain",
        3: "Typical angina",
    },
    "fbs": {1: "Yes (> 120 mg/dl)", 0: "No"},
    "restecg": {
        0: "Left ventricular hypertrophy",
        1: "Normal",
        2: "ST-T wave abnormality",
    },
    "exang": {1: "Yes", 0: "No"},
    "slope": {0: "Downsloping", 1: "Flat", 2: "Upsloping"},
    "thal": {1: "Fixed defect", 2: "Normal", 3: "Reversible defect"},
}


@dataclass
class ValidationError:
    field: str
    message: str


def validate_patient_input(record: Dict) -> list:
    """Same strict schema at every inference boundary."""
    from .schemas import PatientInput
    from pydantic import ValidationError as SchemaError
    try:
        PatientInput.model_validate(record)
    except SchemaError as exc:
        return [ValidationError(str(e["loc"][0]), e["msg"]) for e in exc.errors()]
    return []


def load_dataset(path=DATA_PATH, profile="heartwise_redistributed_v1") -> pd.DataFrame:
    path = Path(path)
    if profile != "heartwise_redistributed_v1":
        raise ValueError("Unsupported dataset profile; canonical UCI import uses src.audit.canonical_uci")
    if hashlib.sha256(path.read_bytes()).hexdigest() != DATA_HASH:
        raise ValueError("Dataset hash does not match heartwise_redistributed_v1; audit a new profile first")
    df = pd.read_csv(path)
    # Verified against all UCI source fields by src.audit, not inferred from correlations.
    df["target"] = 1 - df["target"]
    return df[(df.ca != 4) & (df.thal != 0)].copy()


def build_preprocessor() -> ColumnTransformer:
    """The exact same preprocessing is applied at train time and predict
    time because both go through this one object, saved inside the
    fitted Pipeline. This is what section 8 of the improvement notes
    ('proper preprocessing pipeline') is about: eliminating any chance of
    train/predict preprocessing drift.

    Categories are passed explicitly (rather than left as 'auto') so that
    every one-hot column always exists regardless of which rows happen to
    land in a given train/CV-fold split — with a rare category (e.g. only
    18 patients have thal=1 out of the whole dataset), a fold that happens
    to miss it entirely would otherwise silently produce one fewer column
    than expected, which breaks anything that depends on a fixed column
    count/order (this is what caused XGBoost's monotonic constraints to
    fail during cross-validation before this was added)."""
    categorical_categories = [
        sorted(CATEGORY_OPTIONS[col].keys()) for col in CATEGORICAL_FEATURES
    ]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(categories=categorical_categories, handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def get_feature_names_out(preprocessor: ColumnTransformer) -> list:
    """Raw sklearn output feature names after the ColumnTransformer has been
    fit (e.g. 'num__age', 'cat__cp_0'). Use prettify_feature_name() to turn
    these into labels a non-technical reader can follow."""
    return list(preprocessor.get_feature_names_out())


def prettify_feature_name(raw_name: str) -> str:
    """'num__age' -> 'Age'; 'cat__cp_0' -> 'Chest Pain Type = Typical angina'.
    Falls back to the raw name if it doesn't match the expected pattern —
    used everywhere a feature name is shown to a user (SHAP/importance
    plots, the API's per-prediction explanation, the Streamlit report)."""
    if raw_name.startswith("num__"):
        col = raw_name[len("num__"):]
        return FEATURE_LABELS.get(col, col)

    if raw_name.startswith("cat__"):
        remainder = raw_name[len("cat__"):]
        # remainder looks like "cp_0" — the column name, then its value code.
        # Match against known columns (longest name first, since e.g. both
        # 'cp' and 'ca' could otherwise ambiguously prefix-match).
        for col in sorted(CATEGORICAL_FEATURES, key=len, reverse=True):
            prefix = f"{col}_"
            if remainder.startswith(prefix):
                value_str = remainder[len(prefix):]
                try:
                    value = int(value_str)
                    value_label = CATEGORY_OPTIONS.get(col, {}).get(value, value_str)
                except ValueError:
                    value_label = value_str
                return f"{FEATURE_LABELS.get(col, col)} = {value_label}"
        return remainder

    return raw_name


def prettify_feature_names(raw_names: list) -> list:
    return [prettify_feature_name(n) for n in raw_names]


def make_pipeline(model) -> Pipeline:
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("classifier", model)])
