import os
import pandas as pd
import spacy
from tqdm import tqdm

from features.extractor import extract_features
# from features.bert_surprisal import sentence_surprisal

# NER currently disabled for speed
nlp = spacy.load("en_core_web_sm", disable=["ner"])

DATA_FOLDER = "data/onestopenglish"

difficulty_map = {
    "Elementary": 2,
    "Intermediate": 6,
    "Advanced": 10
}

rows = []

files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")]

for filename in tqdm(files, desc="Processing files"):

    path = os.path.join(DATA_FOLDER, filename)

    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp1252")

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Rows in {filename}", leave=False):

        for level_col, difficulty in difficulty_map.items():

            if level_col not in df.columns:
                continue

            text = row[level_col]

            if pd.isna(text):
                continue

            # parse once
            doc = nlp(str(text))

            for sent in doc.sents:

                if len(sent) < 3:
                    continue

                feats = extract_features(sent)

                rows.append({
                    "text": sent.text.strip(),
                    **feats,
                    "difficulty": difficulty
                })

print("Total sentences:", len(rows))

final_df = pd.DataFrame(rows)
final_df.to_csv("data/features.csv", index=False)

print("features.csv rebuilt successfully.")
