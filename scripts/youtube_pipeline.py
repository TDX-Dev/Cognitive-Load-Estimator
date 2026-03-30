import re
from youtube_transcript_api import YouTubeTranscriptApi
from models.ranking_predict import score_sentence

from youtube_transcript_api.formatters import JSONFormatter

import spacy

nlp = spacy.load("en_core_web_sm", disable=["ner"])


from urllib.parse import urlparse, parse_qs

def extract_video_id(url: str):
    parsed = urlparse(url)

    if parsed.hostname in ["www.youtube.com", "youtube.com"]:
        return parse_qs(parsed.query).get("v", [None])[0]

    if parsed.hostname == "youtu.be":
        return parsed.path.lstrip("/")

    if "watch?v=" in url:
        return url.split("watch?v=")[-1].split("&")[0]

    if len(url) == 11:
        return url

    return None


def get_transcript(video_id):
    ytt = YouTubeTranscriptApi()
    transcript = ytt.fetch(video_id)
    return transcript

def analyze_video(video_url):

    transcript = get_transcript(extract_video_id(video_url))

    results = []

    for segment in transcript:

        text = segment.text
        start = segment.start
        duration = segment.duration


        doc = nlp(text)

        for sent in doc.sents:
            if len(sent.text.strip()) < 5:
                continue

            score = score_sentence(sent)

            results.append({
                "sentence": sent.text.strip(),
                "score": float(score),
                "start": start,
                "end": start + duration
            })

    return results