import os
import pandas as pd
import spacy
from tqdm import tqdm
from features.extractor import extract_features


nlp = spacy.load("en_core_web_sm", disable=["ner"])

DATA_FOLDER = "data/onestopenglish"

rows = []

files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")]

for filename in tqdm(files, desc="Processing files"):

    path = os.path.join(DATA_FOLDER, filename)

    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp1252")

    # checking if required columns exist
    if not all(col in df.columns for col in ["Elementary", "Intermediate", "Advanced"]):
        continue

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"Rows in {filename}",
                       leave=False):

        E_text = str(row["Elementary"])
        I_text = str(row["Intermediate"])
        A_text = str(row["Advanced"])

        # Parse once per level
        E_doc = nlp(E_text)
        I_doc = nlp(I_text)
        A_doc = nlp(A_text)

        E_sents = list(E_doc.sents)
        I_sents = list(I_doc.sents)
        A_sents = list(A_doc.sents)

        min_len = min(len(E_sents), len(I_sents), len(A_sents))

        # Sentence alignment progress
        for i in range(min_len):

            E_feat = extract_features(E_sents[i])
            I_feat = extract_features(I_sents[i])
            A_feat = extract_features(A_sents[i])

            rows.append({
                "E": E_feat,
                "I": I_feat,
                "A": A_feat
            })

print("Total aligned sentence triples:", len(rows))

pd.to_pickle(rows, "data/ranking_dataset.pkl")

print("Ranking dataset built successfully.")
