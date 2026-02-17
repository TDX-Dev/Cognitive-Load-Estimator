import spacy
from models.ranking_predict import score_text
from visualization.heatmap import generate_text_heatmap

# Load spaCy once
nlp = spacy.load("en_core_web_sm", disable=["ner"])


def analyze_text(text):
    """
    Returns list of:
    {
        "sentence": str,
        "score": float (0–10 normalized)
    }
    """
    results = []

    sentence_scores = score_text(text)

    for sentence, score in sentence_scores:
        results.append({
            "sentence": sentence,
            "score": float(score)
        })

    return results


if __name__ == "__main__":

    sample_text = (
        "The cat slept on the sofa."
        "Students learn quickly when the material is simple and engaging." 
        "However, understanding complex academic literature requires sustained cognitive effort and analytical reasoning."
        "Notwithstanding the theoretical elegance of the framework, its practical implementation remains fraught with ambiguity." 
        "Water boils at one hundred degrees Celsius."
        "The bank can run light set."
    )

    results = analyze_text(sample_text)

    html = generate_text_heatmap(results)

    with open("heatmap_output.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("Heatmap saved to heatmap_output.html")
