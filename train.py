"""
train.py
Training pipeline for GDM risk prediction.

Run from project root:
    python train.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).resolve().parent
DATA   = ROOT / "data" / "dataset_en.csv"
MODELS = ROOT / "models"

# ── Columns ───────────────────────────────────────────────────────────────────
TARGET   = "Gestational_Diabetes"

FEATURES = [
    "Age", "Previous_Pregnancies", "Previous_C_Sections",
    "Diastolic_BP_mmHg", "Serum_Insulin_2h", "BMI",
]

VALID_RANGES = {
    "Age":                  (14, 55),
    "Previous_Pregnancies": (0,  20),
    "Previous_C_Sections":  (0,  12),
    "Diastolic_BP_mmHg":    (40, 130),
    "Serum_Insulin_2h":     (0,  400),
    "BMI":                  (14, 60),
}

CLIP_COLS = ["Age", "Diastolic_BP_mmHg", "Serum_Insulin_2h", "BMI"]


# ── Cleaning ──────────────────────────────────────────────────────────────────
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop rows with missing target
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

    # Replace capped insulin sensor value
    df["Serum_Insulin_2h"] = df["Serum_Insulin_2h"].replace(320.0, np.nan)

    # Clinical range → NaN
    for col, (lo, hi) in VALID_RANGES.items():
        mask = (df[col] < lo) | (df[col] > hi)
        df.loc[mask, col] = np.nan

    # Logical fix: C_sections cannot exceed pregnancies
    mask = df["Previous_C_Sections"] > df["Previous_Pregnancies"]
    df.loc[mask, "Previous_C_Sections"] = df.loc[mask, "Previous_Pregnancies"]

    # IQR clip
    for col in CLIP_COLS:
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo  = max(q1 - 1.5 * iqr, VALID_RANGES[col][0])
        hi  = min(q3 + 1.5 * iqr, VALID_RANGES[col][1])
        df[col] = df[col].clip(lo, hi)

    return df


# ── Preprocessor ──────────────────────────────────────────────────────────────
def build_preprocessor() -> ColumnTransformer:
    continuous_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    return ColumnTransformer([
        ("continuous", continuous_pipe, FEATURES),
    ])


# ── Train ─────────────────────────────────────────────────────────────────────
def train() -> None:
    print(f"Loading {DATA}")
    df = pd.read_csv(DATA)
    df = clean(df)
    print(f"Clean shape: {df.shape}")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Train balance:\n{y_train.value_counts(normalize=True).round(3)}")

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier",   LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])

    param_grid = [{
        "classifier":    [LogisticRegression(max_iter=2000, class_weight="balanced")],
        "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
    }]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        pipeline, param_grid,
        scoring="roc_auc", cv=cv,
        n_jobs=-1, verbose=1
    )

    print("\nRunning GridSearchCV...")
    grid.fit(X_train, y_train)

    print(f"\nBest params : {grid.best_params_}")
    print(f"Best CV AUC : {grid.best_score_:.4f}")

    # Final evaluation on test set
    best    = grid.best_estimator_
    y_prob  = best.predict_proba(X_test)[:, 1]
    y_pred  = best.predict(X_test)

    test_auc = roc_auc_score(y_test, y_prob)
    print(f"\nTest ROC-AUC: {test_auc:.4f}")
    print(classification_report(y_test, y_pred))

    # Save model package
    MODELS.mkdir(exist_ok=True)
    package = {
        "pipeline":  best,
        "features":  FEATURES,
        "target":    TARGET,
        "threshold": 0.5,
        "metrics": {
            "test_roc_auc":  round(test_auc, 4),
            "best_cv_auc":   round(float(grid.best_score_), 4),
            "best_params":   str(grid.best_params_),
        }
    }
    out = MODELS / "model_gdm.joblib"
    joblib.dump(package, out)
    print(f"\nModel saved: {out}")


if __name__ == "__main__":
    train()