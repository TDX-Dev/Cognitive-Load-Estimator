import numpy as np

def score_to_color(score):
    # score is 0–10
    normalized = score / 10

    red = int(255 * normalized)
    green = int(255 * (1 - normalized))
    blue = 120

    return f"rgb({red},{green},{blue})"



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
        total_tokens = sum(len(item["sentence"].split()) for item in results)

        weighted_sum = sum(
            item["score"] * len(item["sentence"].split())
            for item in results
        )

        final_score = round(weighted_sum / total_tokens, 2)

    else:
        final_score = 0

    html += f"""
    <hr>
    <h3 style="color:white">Overall Cognitive Load Score: {final_score} / 10</h3>
    """

    return html

def generate_timeline_heatmap(results, total_duration):

    html = """
    <div style="
        width:100%;
        height:30px;
        display:flex;
        border-radius:6px;
        overflow:hidden;
    ">
    """

    for item in results:
        start = item["start"]
        end = item["end"]
        score = item["score"]

        width_percent = ((end - start) / total_duration) * 100

        color = score_to_color(score)

        html += f"""
        <div style="
            width:{width_percent}%;
            background:{color};
        "></div>
        """

    html += "</div>"

    return html