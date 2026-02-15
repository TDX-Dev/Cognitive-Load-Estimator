import spacy

nlp = spacy.load("en_core_web_sm")

CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}

def lexical_density(text):
    """
    ratio of content words to total words
    higher = harder text
    """
    doc = nlp(text)
    content_words = 0
    total_words = 0
    for t in doc:
        if t.pos_ in CONTENT_POS:
            content_words += 1
        if t.is_alpha:
            total_words += 1
    if total_words == 0:
        return 0
    
    return content_words / total_words

def avg_sentence_length(text):
    """
    average number of words per sentence
    longer = harder
    """

    doc = nlp(text)

    sentences = list(doc.sents)

    if len(sentences) == 0:
        return 0
    
    lengths = []
    for sent in sentences:
        leng = 0
        for t in sent:
            if t.is_alpha:
                leng += 1
        lengths.append(leng)
    
    return sum(lengths) / len(lengths)

if __name__ == "__main__":
    sample = (
        "Students learn quickly. "
        "However, understanding complex academic material requires effort."
    )
    print("Lexical density:", lexical_density(sample))
    print("Avg sentence length:", avg_sentence_length(sample))