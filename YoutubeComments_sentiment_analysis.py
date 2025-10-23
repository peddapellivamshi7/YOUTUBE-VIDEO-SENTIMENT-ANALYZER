import streamlit as st
from googleapiclient.discovery import build
from transformers import pipeline
import random
import re
from urllib.parse import urlparse, parse_qs
import pandas as pd

# --- Multilingual sentiment model ---
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

# --- YouTube API key ---
YOUTUBE_API_KEY = "AIzaSyAcwE1DCGYtBYVG9cvu0O5oX35vIA13xFU"

# --- Extract Video ID ---
def extract_video_id(url_or_id: str) -> str:
    if re.match(r"^[A-Za-z0-9_-]{10,}$", url_or_id):
        return url_or_id
    parsed = urlparse(url_or_id)
    qs = parse_qs(parsed.query)
    if "v" in qs:
        return qs["v"][0]
    if parsed.netloc.endswith("youtu.be"):
        return parsed.path.lstrip("/")
    raise ValueError("Could not parse video id from input: " + str(url_or_id))

# --- Fetch comments ---
def get_comments(video_id, max_comments=100):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY, cache_discovery=False)
    comments = []
    next_page_token = None
    while True:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            textFormat="plainText",
            pageToken=next_page_token
        ).execute()
        for item in response.get("items", []):
            comments.append(item["snippet"]["topLevelComment"]["snippet"]["textDisplay"])
            if len(comments) >= max_comments:
                return comments
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
    return comments

# --- Analyze overall sentiment ---
def analyze_video_sentiment(comments, sample_size=50):
    if len(comments) > sample_size:
        comments = random.sample(comments, sample_size)  # sample subset
    results = []
    for c in comments:
        result = sentiment_analyzer(c[:512])[0]  # truncate to 512 tokens
        star = int(result["label"][0])
        if star <= 2:
            results.append("Negative")
        elif star == 3:
            results.append("Neutral")
        else:
            results.append("Positive")
    # Aggregate overall sentiment
    sentiment_counts = pd.Series(results).value_counts()
    overall = sentiment_counts.idxmax()  # sentiment with most votes
    return overall

# --- Streamlit App ---
st.title("🎬 YouTube Video Sentiment Analysis (Multilingual)")

video_url = st.text_input("Enter YouTube Video URL or ID:")
max_comments = st.slider("Max Comments to Fetch", 50, 500, 200)
sample_size = st.slider("Number of Comments to Analyze", 10, 100, 50)
start_analysis = st.button("Analyze Video")

if start_analysis:
    if not video_url:
        st.error("Enter a valid YouTube URL or ID.")
    else:
        try:
            video_id = extract_video_id(video_url.strip())
            st.info(f"Fetching comments for video ID: {video_id}...")
            comments = get_comments(video_id, max_comments=max_comments)
            if not comments:
                st.warning("No comments found for this video.")
            else:
                overall_sentiment = analyze_video_sentiment(comments, sample_size=sample_size)
                st.subheader("🎯 Video-Level Sentiment")
                st.write(f"**Overall Sentiment:** {overall_sentiment}")
        except Exception as e:
            st.error(f"Error: {e}")
