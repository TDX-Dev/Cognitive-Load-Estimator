import joblib
import pandas as pd

bundle = joblib.load("models/difficulty.pkl")

model = bundle["model"]
feature_names = bundle["feature_names"]

def predict(features_dict):
    df = pd.DataFrame([features_dict], columns=feature_names)
    prediction = model.predict(df)[0]
    return float(prediction)