import spacy
import pandas as pd
import joblib

from features.extractor import extract_features
from features.bert_surprisal import sentence_surprisal

# Load spaCy once
nlp = spacy.load("en_core_web_sm")

# Load trained model bundle
bundle = joblib.load("models/difficulty.pkl")
model = bundle["model"]
feature_names = bundle["feature_names"]


def predict(features_dict):
    """
    features_dict: dictionary with correct feature names
    returns predicted class
    """
    df = pd.DataFrame([features_dict], columns=feature_names)
    return model.predict(df)[0]


def analyze_text(text):
    doc = nlp(text)

    results = []

    for sent in doc.sents:
        sentence_text = sent.text.strip()
        if not sentence_text:
            continue

        feats = extract_features(sentence_text)
        surprisal = sentence_surprisal(sentence_text)
        feats["surprisal"] = surprisal

        for col in feature_names:
            if col not in feats:
                feats[col] = 0

        score = predict(feats)

        results.append({
            "sentence": sentence_text,
            "score": score
        })

    return results



if __name__ == "__main__":
    sample_text = (
        "Students learn quickly. "
        "However, understanding complex academic material requires significant cognitive effort."
    )

    results = analyze_text(sample_text)

    from visualization.heatmap import generate_text_heatmap

    html = generate_text_heatmap(results)

    with open("heatmap_output.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("Heatmap saved to heatmap_output.html")


