"""
feature_extractor
"""

from features.morphological import lexical_density, avg_sentence_length, type_token_ratio, avg_word_length
from features.syntax import average_dependency_distance, avg_tree_depth, clause_count
from features.semantic import polysemy_score

def extract_features(doc):
    return {
        "lexical_density": lexical_density(doc),
        "sentence_length": avg_sentence_length(doc),
        "avg_word_length": avg_word_length(doc),
        "type_token_ratio": type_token_ratio(doc),
        "dep_distance": average_dependency_distance(doc),
        "tree_depth": avg_tree_depth(doc),
        "clause_count": clause_count(doc),
        "polysemy": polysemy_score(doc)
    }

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