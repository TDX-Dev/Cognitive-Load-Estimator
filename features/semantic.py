from nltk.corpus import wordnet as wn

def polysemy_score(doc):
    counts = []

    for token in doc:
        if token.is_alpha:
            synsets = wn.synsets(token.text)
            if synsets:
                counts.append(len(synsets))

    if not counts:
        return 0

    return sum(counts) / len(counts)

if __name__ == "__main__":
    easy = "The cat sleeps."
    ambiguous = "The bank can run light set."

    print("Easy polysemy:", polysemy_score(easy))
    print("Ambiguous polysemy:", polysemy_score(ambiguous))