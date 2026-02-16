def average_dependency_distance(doc):
    distances = []

    for token in doc:
        if token.head != token:
            distances.append(abs(token.i - token.head.i))

    if not distances:
        return 0

    return sum(distances) / len(distances)

def clause_count(doc):
    count = 0
    for token in doc:
        if token.dep_ in ["advcl", "ccomp", "xcomp", "relcl"]:
            count += 1
    return count

def tree_depth(token):
    children = list(token.children)
    if not children:
        return 1
    return 1 + max(tree_depth(child) for child in children)

def avg_tree_depth(doc):
    roots = [t for t in doc if t.head == t]

    if not roots:
        return 0

    depths = [tree_depth(root) for root in roots]
    return sum(depths) / len(depths)

if __name__ == "__main__":
    easy = "The cat sat on the mat."
    hard = "Students who attempt to understand deeply nested academic sentences often struggle significantly."

    print("Easy distance:", average_dependency_distance(easy))
    print("Hard distance:", average_dependency_distance(hard))

    print("Easy depth:", avg_tree_depth(easy))
    print("Hard depth:", avg_tree_depth(hard))

# comment for testing