import spacy

nlp = spacy.load("en_core_web_sm")

def average_dependency_distance(text):
    """
    average | token idex - head index |
    higher = more complex syntax
    """
    doc = nlp(text)

    distances = []

    for token in doc:
        if token.head != token:
            distances.append(abs(token.i - token.head.i))
    if len(distances) == 0:
        return 0
    
    return sum(distances) / len(distances)


def tree_depth(token):
    """recursive"""

    children = list(token.children)

    if not children:
        return 1
    
    return 1 + max(tree_depth(child) for child in children)

def avg_tree_depth(text):
    """
    average parse tree depth
    deeper = more nester = harder
    """

    doc = nlp(text)

    roots = [t for t in doc if t.head == t]

    depths = [tree_depth(root) for root in roots]

    if len(depths) == 0:
        return 0
    
    return sum(depths) / len(depths)

if __name__ == "__main__":
    easy = "The cat sat on the mat."
    hard = "Students who attempt to understand deeply nested academic sentences often struggle significantly."

    print("Easy distance:", average_dependency_distance(easy))
    print("Hard distance:", average_dependency_distance(hard))

    print("Easy depth:", avg_tree_depth(easy))
    print("Hard depth:", avg_tree_depth(hard))