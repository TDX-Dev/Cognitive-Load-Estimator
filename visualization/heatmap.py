import numpy as np

def score_to_color(score):
    """
    Convert difficulty score (0,1,2) into color.
    0 = Easy (Green)
    1 = Medium (Yellow)
    2 = Hard (Red)
    """

    colors = {
        0: "#b7f7b7",   # light green
        1: "#fff3a3",   # yellow
        2: "#ffb3b3"    # light red
    }

    return colors.get(score, "#ffffff")


def generate_text_heatmap(results):
    """
    results = list of {"sentence": str, "score": int}
    returns HTML string
    """

    html = ""
    numeric_scores = []

    for item in results:
        sentence = item["sentence"]
        score = item["score"]
        numeric_scores.append(score)

        color = score_to_color(score)

        html += f"""
        <span style="
            background-color:{color};
            padding:4px;
            border-radius:4px;
            margin:2px;
            display:inline-block;
        ">
            {sentence}
        </span>
        """

    # ---- Final score out of 10 ----
    # normalize (0–2 scale → 0–10 scale)
    if len(numeric_scores) > 0:
        avg = np.mean(numeric_scores)
        final_score = round((avg / 2) * 10, 2)
    else:
        final_score = 0

    html += f"""
    <hr>
    <h3>Overall Cognitive Load Score: {final_score} / 10</h3>
    """

    return html

