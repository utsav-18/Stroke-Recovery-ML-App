import joblib
import numpy as np

model = joblib.load("model/stroke_model.pkl")

LABEL_MAP = {
    0: "NOT IMPROVED",
    1: "IMPROVING",
    2: "IMPROVED"
}

def predict_recovery(features):
    features = np.array(features).reshape(1, -1)
    pred = model.predict(features)[0]
    return LABEL_MAP.get(int(pred), "UNKNOWN")
