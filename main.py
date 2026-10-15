import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from xgboost import XGBClassifier, XGBRegressor

# ── Load Models ───────────────────────────────────────────────────────────
clf = XGBClassifier()
clf.load_model('model_clf.json')
reg = XGBRegressor()
reg.load_model('model_reg.json')
scaler_clf  = joblib.load('scaler_clf.pkl')
scaler_reg  = joblib.load('scaler_reg.pkl')
imputer     = joblib.load('imputer.pkl')

# ── App Setup ─────────────────────────────────────────────────────────────
app = FastAPI(title="Aqua-Epoch Water Quality API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow frontend to call this API
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Input Schema ──────────────────────────────────────────────────────────
class WaterInput(BaseModel):
    Turbidity : float
    Hardness  : float
    TDS       : float

# ── Feature Order (must match training) ──────────────────────────────────
# [Hardness, TDS, Chloramines, Sulfate, Conductivity,
#  Organic_carbon, Trihalomethanes, Turbidity, Calcium,
#  Magnesium, Sodium, Potassium, Chloride, Nitrate,
#  Fluoride, Iron, Alkalinity]

MEDIAN_VALUES = [
    196.0,   # Hardness
    500.0,   # TDS
    7.1,     # Chloramines
    250.0,   # Sulfate
    426.0,   # Conductivity
    14.0,    # Organic_carbon
    66.0,    # Trihalomethanes
    3.9,     # Turbidity
    75.0,    # Calcium
    28.0,    # Magnesium
    50.0,    # Sodium
    10.0,    # Potassium
    150.0,   # Chloride
    20.0,    # Nitrate
    0.9,     # Fluoride
    0.2,     # Iron
    120.0    # Alkalinity
]

# ── Predict Endpoint ──────────────────────────────────────────────────────
@app.post("/predict")
def predict(data: WaterInput):

    # Start with median values for all 17 features
    features = MEDIAN_VALUES.copy()

    # Override with user provided values
    features[0] = data.Hardness    # index 0 → Hardness
    features[1] = data.TDS         # index 1 → TDS
    features[7] = data.Turbidity   # index 7 → Turbidity

    # Convert to numpy array
    sample = np.array([features])

    # Scale for classifier
    sample_clf = scaler_clf.transform(sample)

    # Scale for regressor
    sample_reg = scaler_reg.transform(sample)

    # Predict
    potability = int(clf.predict(sample_clf)[0])
    ph         = float(reg.predict(sample_reg)[0])

    return {
        "safe"        : potability == 1,
        "potability"  : potability,
        "predicted_ph": round(ph, 2),
        "message"     : "Safe to drink ✅" if potability == 1 else "Not safe to drink ❌"
    }

# ── Health Check ──────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "Aqua-Epoch API is running ✅"}
