import numpy as np


# ----------- Improved Color Mapping -----------

def score_to_color(score):
    """
    Maps 0–10 score to smooth gradient:
    Blue → Teal → Yellow → Orange → Red
    """
    normalized = max(0, min(1, score / 10))

    # HSV-style gradient interpolation
    # Hue from 210 (blue) to 0 (red)
    hue = (1 - normalized) * 210
    saturation = 85
    lightness = 55

    return f"hsl({hue}, {saturation}%, {lightness}%)"


# ----------- Text Heatmap -----------

def generate_text_heatmap(results):
    """
    results = list of {"sentence": str, "score": float}
    returns styled HTML string
    """

    html = """
    <div style="
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        line-height: 1.8;
        font-size: 16px;
    ">
    """

    numeric_scores = []

    for item in results:
        sentence = item["sentence"]
        score = float(item["score"])
        numeric_scores.append(score)

        color = score_to_color(score)

        html += f"""
        <span title="Score: {round(score,2)} / 10" style="
            background:{color};
            padding:6px 10px;
            border-radius:10px;
            margin:4px 4px;
            display:inline-block;
            color:white;
            font-weight:500;
            box-shadow:0 2px 6px rgba(0,0,0,0.15);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        " 
        onmouseover="this.style.transform='scale(1.05)'; this.style.boxShadow='0 4px 12px rgba(0,0,0,0.25)';"
        onmouseout="this.style.transform='scale(1)'; this.style.boxShadow='0 2px 6px rgba(0,0,0,0.15)';"
        >
            {sentence}
        </span>
        """

    # ---- Weighted Final Score ----
    if len(numeric_scores) > 0:
        total_tokens = sum(len(item["sentence"].split()) for item in results)
        weighted_sum = sum(
            item["score"] * len(item["sentence"].split())
            for item in results
        )
        final_score = round(weighted_sum / total_tokens, 2)
    else:
        final_score = 0

    final_color = score_to_color(final_score)

    html += f"""
        <div style="margin-top:30px; text-align:center;">
            <div style="
                font-size:14px;
                color:#888;
                margin-bottom:6px;
                letter-spacing:1px;
                text-transform:uppercase;
            ">
                Overall Cognitive Load
            </div>
            <div style="
                font-size:28px;
                font-weight:700;
                color:{final_color};
            ">
                {final_score} / 10
            </div>
        </div>
    </div>
    """

    return html


# ----------- Timeline Heatmap -----------

def generate_timeline_heatmap(results, total_duration):
    """
    results = list of {"start": float, "end": float, "score": float}
    """

    html = """
    <div style="
        width:100%;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    ">
        <div style="
            width:100%;
            height:32px;
            display:flex;
            border-radius:12px;
            overflow:hidden;
            box-shadow:0 4px 14px rgba(0,0,0,0.15);
        ">
    """

    for item in results:
        start = item["start"]
        end = item["end"]
        score = float(item["score"])

        width_percent = ((end - start) / total_duration) * 100
        color = score_to_color(score)

        html += f"""
        <div title="Score: {round(score,2)} / 10"
            style="
                width:{width_percent}%;
                background:{color};
                transition:filter 0.2s ease;
            "
            onmouseover="this.style.filter='brightness(1.2)';"
            onmouseout="this.style.filter='brightness(1)';"
        ></div>
        """

    html += """
        </div>
        <div style="
            display:flex;
            justify-content:space-between;
            font-size:12px;
            color:#888;
            margin-top:6px;
        ">
            <span>0:00</span>
            <span>End</span>
        </div>
    </div>
    """

    return html