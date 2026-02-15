import joblib
import pandas as pd

bundle = joblib.load("models/difficulty.pkl")

model = bundle["model"]
feature_names = bundle["feature_names"]
label_encoder = bundle["label_encoder"]

def predict(features_dict):
    df = pd.DataFrame([features_dict], columns=feature_names)
    numeric_prediction = model.predict(df)[0]
    return numeric_prediction  # keep numeric for heatmap

def predict_label(features_dict):
    df = pd.DataFrame([features_dict], columns=feature_names)
    numeric_prediction = model.predict(df)[0]
    return label_encoder.inverse_transform([numeric_prediction])[0]
