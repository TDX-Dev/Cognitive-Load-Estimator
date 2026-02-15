"""
feature_extractor
"""

from features.morphological import lexical_density, avg_sentence_length
from features.syntax import average_dependency_distance, avg_tree_depth
from features.semantic import polysemy_score

def extract_features(text):
    """
    Combine all feature modules into one dictionary
    text -> dict with numeric features sorta like json
    """

    features = {
        "lexical_density": lexical_density(text),
        "avg_sentence_length": avg_sentence_length(text),
        "dep_distance": average_dependency_distance(text),
        "tree_depth": avg_tree_depth(text),
        "polysemy": polysemy_score(text)
    }

    return features

if __name__ == "__main__":
    easy = "The dog runs."
    hard = "Students who attempt to understand deeply nested academic sentences often struggle significantly."


    feasy, fhard = extract_features(easy), extract_features(hard)
    print("EASY")
    for k, v in feasy.items():
        print(k, ":", v)
    print("HARD")
    for k, v in fhard.items():
        print(k, ":", v)