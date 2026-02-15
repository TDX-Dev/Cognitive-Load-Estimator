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
    """
    Splits text into sentences,
    extracts features per sentence,
    adds BERT surprisal,
    predicts difficulty per sentence.
    Returns list of scores.
    """
    doc = nlp(text)
    scores = []

    for sent in doc.sents:
        sentence_text = sent.text.strip()

        if not sentence_text:
            continue

        # 1️⃣ Extract Person A features
        feats = extract_features(sentence_text)

        # 2️⃣ Add BERT surprisal
        surprisal = sentence_surprisal(sentence_text)
        feats["surprisal"] = surprisal

        # 3️⃣ Ensure all expected features exist
        # (important for safety)
        for col in feature_names:
            if col not in feats:
                feats[col] = 0

        # 4️⃣ Predict
        score = predict(feats)
        scores.append(score)

    return scores


if __name__ == "__main__":
    sample_text = (
        "Students learn quickly. "
        "However, understanding complex academic material requires significant cognitive effort."
    )

    results = analyze_text(sample_text)

    print("Sentence difficulty scores:")
    print(results)

    from visualization.heatmap import plot_heatmap
    plot_heatmap(results)

