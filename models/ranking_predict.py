import torch
import torch.nn as nn
import joblib
import spacy
from features.extractor import extract_features

# load spaCy
nlp = spacy.load("en_core_web_sm", disable=["ner"])

# load saved model bundle
bundle = torch.load("models/ranking_model.pt")

min_score = bundle["min_score"]
max_score = bundle["max_score"]

scaler = joblib.load("models/ranking_scaler.pkl")

# define model architecture
class RankModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

# initialize model
input_dim = scaler.mean_.shape[0]
model = RankModel(input_dim)
model.load_state_dict(bundle["model_state"])
model.eval()

def score_sentence(sentence):
    feats = extract_features(sentence)
    vec = list(feats.values())

    vec_scaled = scaler.transform([vec])
    tensor = torch.tensor(vec_scaled, dtype=torch.float32)

    with torch.no_grad():
        raw_score = model(tensor).item()

    normalized = 10 * (raw_score - min_score) / (max_score - min_score)

    return max(0, min(10, normalized))


def score_text(text):
    doc = nlp(text)
    results = []

    for sent in doc.sents:
        if len(sent) < 3:
            continue

        score = score_sentence(sent)
        results.append((sent.text.strip(), score))

    return results
