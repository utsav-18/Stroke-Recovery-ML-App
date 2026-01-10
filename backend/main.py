from fastapi import FastAPI
from vision.capture import capture_upper_limb
from features.extractor import extract_features
from inference.predict import predict_recovery

app = FastAPI()

@app.get("/")
def root():
    return {"status": "Stroke Recovery ML Backend Running"}

@app.get("/assess")
def assess():
    try:
        angles = capture_upper_limb(seconds=5)

        if angles is None or len(angles) < 10:
            return {
                "error": "No clear upper-limb movement detected",
                "angles_captured": int(len(angles)) if angles else 0
            }

        features = extract_features(angles)

        prediction = predict_recovery(features)

        return {
            "prediction": str(prediction),  # ✅ convert numpy → str
            "features": [float(x) for x in features],  # ✅ numpy → float
            "frames_used": int(len(angles))
        }

    except Exception as e:
        return {
            "error": "Internal ML pipeline failure",
            "details": str(e)
        }
