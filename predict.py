from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent
MODEL_PATH    = ROOT / "models" / "model_gdm.joblib"

# ── Cache ─────────────────────────────────────────────────────────────────────
_MODEL_CACHE: Optional[Dict[str, Any]] = None


def load_model(path: Path = MODEL_PATH) -> Dict[str, Any]:
    """Load model from disk once, cache in memory."""
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}\nRun train.py first."
        )
    _MODEL_CACHE = joblib.load(path)
    return _MODEL_CACHE


def predict_gdm(
    Age: float,
    Previous_Pregnancies: float,
    Previous_C_Sections: float,
    Diastolic_BP_mmHg: float,
    Serum_Insulin_2h: float,
    BMI: float,
) -> Dict[str, Any]:
    """
    Predict gestational diabetes risk for a single patient.

    Returns:
    {
        "prediction":  0 or 1,
        "risk_score":  float between 0 and 1,
        "label":       "High Risk" or "Low Risk",
        "threshold":   float
    }
    """
    package   = load_model()
    pipeline  = package["pipeline"]
    features  = package["features"]
    threshold = package.get("threshold", 0.5)

    patient = {
        "Age":                  Age,
        "Previous_Pregnancies": Previous_Pregnancies,
        "Previous_C_Sections":  Previous_C_Sections,
        "Diastolic_BP_mmHg":    Diastolic_BP_mmHg,
        "Serum_Insulin_2h":     Serum_Insulin_2h,
        "BMI":                  BMI,
    }

    input_df    = pd.DataFrame([patient], columns=features)
    risk_score  = float(pipeline.predict_proba(input_df)[0, 1])
    prediction  = int(risk_score >= threshold)

    return {
        "prediction": prediction,
        "risk_score": round(risk_score, 4),
        "label":      "High Risk" if prediction == 1 else "Low Risk",
        "threshold":  threshold,
    }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = predict_gdm(
        Age=35,
        Previous_Pregnancies=2,
        Previous_C_Sections=0,
        Diastolic_BP_mmHg=80,
        Serum_Insulin_2h=180,
        BMI=32,
    )
    print(result)