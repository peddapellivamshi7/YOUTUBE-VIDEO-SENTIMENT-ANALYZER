import streamlit as st
import pandas as pd
from googleapiclient.discovery import build
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import time
import random
import re
from urllib.parse import urlparse, parse_qs
import os
from dotenv import load_dotenv


# Initialize Hugging Face Sentiment Model
# -------------------------------------------------
sentiment_model = pipeline(
    task="sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

# -------------------------------------------------
# YouTube API Key
# -------------------------------------------------
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

# -------------------------------------------------
# Function: Extract Video ID from URL or Plain ID
# -------------------------------------------------
def parse_video_identifier(video_input: str) -> str:
    """Return a valid YouTube video ID from a full URL or short ID."""
    if re.match(r"^[A-Za-z0-9_-]{10,}$", video_input):
        return video_input
    parsed = urlparse(video_input)
    query = parse_qs(parsed.query)
    if "v" in query:
        return query["v"][0]
    if parsed.netloc.endswith("youtu.be"):
        return parsed.path.strip("/")
    raise ValueError(f"Unable to extract video ID from: {video_input}")

# -------------------------------------------------
# Function: Retrieve Comments Using YouTube API
# -------------------------------------------------
def fetch_youtube_comments(video_id: str, limit: int = 100):
    """Fetch up to `limit` top-level comments for a given video."""
    yt_service = build("youtube", "v3", developerKey=API_KEY, cache_discovery=False)
    comments, token = [], None

    while True:
        resp = yt_service.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            textFormat="plainText",
            pageToken=token
        ).execute()

        for item in resp.get("items", []):
            comment_text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment_text)
            if len(comments) >= limit:
                return comments

        token = resp.get("nextPageToken")
        if not token:
            break
    return comments

# -------------------------------------------------
# Function: Run Sentiment Analysis on a Comment
# -------------------------------------------------
def evaluate_sentiment(comment_text: str):
    """Analyze a single comment and return sentiment, emotion, and score."""
    result = sentiment_model(comment_text[:512])[0]
    sentiment_label = result["label"].capitalize()
    confidence = result["score"]

    # Random emotion mapping for visualization
    emotion_mapping = {
        "Positive": ["Joy", "Excitement", "Admiration"],
        "Negative": ["Anger", "Sadness", "Disgust"],
        "Neutral": ["Calm", "Indifferent", "Curiosity"]
    }
    emotion = random.choice(emotion_mapping.get(sentiment_label, ["Undefined"]))
    return sentiment_label, emotion, confidence

# -------------------------------------------------
# Streamlit App UI
# -------------------------------------------------
st.set_page_config(page_title="YouTube Sentiment Analyzer", layout="wide")
st.title("🎥 YouTube Comment Sentiment Dashboard")

# User input
video_link = st.text_input("Enter a YouTube video URL or video ID:")
refresh_delay = st.slider("Auto-refresh interval (seconds)", 30, 300, 120)
comment_limit = st.slider("Maximum comments to fetch", 50, 500, 200)
analyze_btn = st.button("Start Sentiment Analysis")

# -------------------------------------------------
# Analysis Workflow
# -------------------------------------------------
if analyze_btn:
    if not API_KEY:
        st.error("YouTube API key is missing. Please provide a valid key.")
    elif not video_link:
        st.error("Please enter a YouTube URL or video ID first.")
    else:
        try:
            video_id = parse_video_identifier(video_link.strip())
            st.info(f"Processing comments for Video ID: {video_id}")
        except Exception as err:
            st.error(f"Error parsing video ID: {err}")
            st.stop()

        seen_comments = set()
        results_df = pd.DataFrame(columns=["Comment", "Sentiment", "Emotion", "Confidence"])

        # Continuous refresh loop
        while True:
            try:
                fetched_comments = fetch_youtube_comments(video_id, limit=comment_limit)
            except Exception as fetch_err:
                st.error(f"Failed to fetch comments: {fetch_err}")
                break

            # Filter newly fetched comments
            new_entries = [c for c in fetched_comments if c not in seen_comments]

            if not new_entries:
                st.info("No new comments found. Waiting for next refresh cycle...")
            else:
                seen_comments.update(new_entries)
                analyzed = [evaluate_sentiment(c) for c in new_entries]

                new_data = pd.DataFrame({
                    "Comment": new_entries,
                    "Sentiment": [s for s, e, sc in analyzed],
                    "Emotion": [e for s, e, sc in analyzed],
                    "Confidence": [sc for s, e, sc in analyzed]
                })

                results_df = pd.concat([results_df, new_data], ignore_index=True)
                results_df["Index"] = range(1, len(results_df) + 1)

                # --- Display Section ---
                st.subheader("🧩 Latest Sentiment Analysis")
                st.dataframe(new_data)

                # Pie chart visualization
                sentiment_summary = results_df["Sentiment"].value_counts()
                pie_chart = px.pie(
                    names=sentiment_summary.index,
                    values=sentiment_summary.values,
                    title="Sentiment Distribution Overview"
                )
                st.plotly_chart(pie_chart, use_container_width=True)

                # Scatter plot over time
                timeline_plot = px.scatter(
                    results_df,
                    x="Index",
                    y="Sentiment",
                    color="Sentiment",
                    hover_data=["Comment"],
                    title="Sentiment Timeline by Comment Index"
                )
                st.plotly_chart(timeline_plot, use_container_width=True)

                # Top comments
                st.subheader("😊 Most Positive Comments")
                st.write(results_df[results_df["Sentiment"] == "Positive"].head(5)[["Comment", "Emotion"]])

                st.subheader("😠 Most Negative Comments")
                st.write(results_df[results_df["Sentiment"] == "Negative"].head(5)[["Comment", "Emotion"]])

                # Word Cloud
                st.subheader("☁️ Word Cloud of All Comments")
                full_text = " ".join(results_df["Comment"])
                wc = WordCloud(width=900, height=400, background_color="white").generate(full_text)
                plt.figure(figsize=(10, 5))
                plt.imshow(wc, interpolation="bilinear")
                plt.axis("off")
                st.pyplot(plt)

                # Save to CSV
                results_df.to_csv("youtube_sentiment_unique.csv", index=False)

            # Delay for refresh
            time.sleep(refresh_delay)
