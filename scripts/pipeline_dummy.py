import spacy
import numpy as np
from models.predict import predict
from features.bert_surprisal import sentence_surprisal

nlp = spacy.load("en_core_web_sm")


def analyze_text(text):
    doc = nlp(text)
    scores = []

    for sent in doc.sents:
        surprisal = sentence_surprisal(sent.text)

        # combine 4 fake features + real surprisal
        fake_other_features = np.random.rand(4)
        features = list(fake_other_features) + [surprisal]

        score = predict(features)
        scores.append(score)

    return scores


if __name__ == "__main__":
    sample = "This is simple. However, complex academic writing increases cognitive demand."
    print(analyze_text(sample))
