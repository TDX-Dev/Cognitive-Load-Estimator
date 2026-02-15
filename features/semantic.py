import spacy
from nltk.corpus import wordnet as wn

nlp = spacy.load("en_core_web_sm")

def polysemy_score(text):
    """
    average number of meanings per word
    higher = more ambigious = harder
    """
    doc = nlp(text)

    counts = []
    
    for token in doc:
        if token.is_alpha:
            synsets = wn.synsets(token.text)
            if synsets:
                counts.append(len(synsets))
    
    if len(counts) == 0:
        return 0
    
    return sum(counts) / len(counts)

if __name__ == "__main__":
    easy = "The cat sleeps."
    ambiguous = "The bank can run light set."

    print("Easy polysemy:", polysemy_score(easy))
    print("Ambiguous polysemy:", polysemy_score(ambiguous))