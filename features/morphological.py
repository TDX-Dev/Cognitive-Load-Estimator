CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}

def type_token_ratio(doc):
    words = [t.text.lower() for t in doc if t.is_alpha]
    if not words:
        return 0
    return len(set(words)) / len(words)

def lexical_density(doc):
    content_words = 0
    total_words = 0

    for token in doc:
        if token.is_alpha:
            total_words += 1
            if token.pos_ in CONTENT_POS:
                content_words += 1

    if total_words == 0:
        return 0

    return content_words / total_words

def avg_sentence_length(doc):
    words = [t for t in doc if t.is_alpha]
    return len(words)

def avg_word_length(doc):
    words = [t.text for t in doc if t.is_alpha]
    if not words:
        return 0
    return sum(len(w) for w in words) / len(words)

if __name__ == "__main__":
    sample = (
        "Students learn quickly. "
        "However, understanding complex academic material requires effort."
    )
    print("Lexical density:", lexical_density(sample))
    print("Avg sentence length:", avg_sentence_length(sample))