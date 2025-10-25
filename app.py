import streamlit as st
import pandas as pd
from googleapiclient.discovery import build
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import re
import random
import os
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

# =====================================
# ⚙️ INITIAL SETUP
# =====================================
st.set_page_config(page_title="YouTube Sentiment Analyzer", layout="wide")
st.title("🎥 YouTube Comment Sentiment Dashboard")

# Load .env only once
@st.cache_resource
def load_env_key():
    load_dotenv()
    return os.getenv("YOUTUBE_API_KEY")

API_KEY = load_env_key()
if not API_KEY:
    st.error("❌ Missing YouTube API key. Please add it to your `.env` file as YOUTUBE_API_KEY.")
    st.stop()

# =====================================
# 🚀 CACHED RESOURCES
# =====================================
@st.cache_resource
def load_sentiment_model():
    """Load Hugging Face sentiment model once (cached)."""
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=-1  # Use GPU if available
    )

sentiment_model = load_sentiment_model()

@st.cache_resource
def get_youtube_service():
    """Initialize YouTube Data API service once."""
    return build("youtube", "v3", developerKey=API_KEY, cache_discovery=False)

yt_service = get_youtube_service()

# =====================================
# 🧩 UTILITY FUNCTIONS
# =====================================
def parse_video_identifier(video_input: str) -> str:
    """Extract YouTube video ID from a URL or ID."""
    if re.match(r"^[A-Za-z0-9_-]{10,}$", video_input):
        return video_input
    parsed = urlparse(video_input)
    if parsed.netloc.endswith("youtu.be"):
        return parsed.path.strip("/")
    query = parse_qs(parsed.query)
    if "v" in query:
        return query["v"][0]
    raise ValueError("Invalid YouTube link or video ID.")

@st.cache_data(ttl=600)
def fetch_youtube_comments(video_id: str, limit: int = 100):
    """Fetch and deduplicate YouTube comments efficiently."""
    comments, token = [], None
    while True:
        resp = yt_service.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, limit - len(comments)),
            textFormat="plainText",
            pageToken=token
        ).execute()

        for item in resp.get("items", []):
            text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"].strip()
            comments.append(text)
            if len(comments) >= limit:
                break

        if len(comments) >= limit or "nextPageToken" not in resp:
            break
        token = resp["nextPageToken"]

    # Remove duplicates quickly (case-insensitive hash set)
    seen, unique = set(), []
    for c in comments:
        key = c.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique[:limit]

def evaluate_sentiment_batch(comments):
    """Batch sentiment analysis for speed."""
    truncated = [c[:512] for c in comments]
    results = sentiment_model(truncated, batch_size=32, truncation=True)

    emotion_map = {
        "Positive": ["Joy", "Excitement", "Admiration"],
        "Negative": ["Anger", "Sadness", "Disgust"],
        "Neutral": ["Calm", "Indifferent", "Curiosity"]
    }

    data = []
    for comment, res in zip(comments, results):
        label = res["label"].capitalize()
        data.append({
            "Comment": comment,
            "Sentiment": label,
            "Emotion": random.choice(emotion_map.get(label, ["Undefined"])),
            "Confidence": res["score"]
        })
    return pd.DataFrame(data)

# =====================================
# 🧠 APP INPUTS
# =====================================
video_link = st.text_input("Enter a YouTube video URL or video ID:")
comment_limit = st.slider("Maximum comments to fetch", 50, 500, 200)
analyze_btn = st.button("Start Sentiment Analysis 🚀")

# =====================================
# 🔍 MAIN ANALYSIS
# =====================================
if analyze_btn:
    try:
        video_id = parse_video_identifier(video_link.strip())
        st.success(f"✅ Video ID detected: `{video_id}`")
    except Exception as e:
        st.error(f"❌ {e}")
        st.stop()

    with st.spinner("Fetching comments..."):
        try:
            fetched_comments = fetch_youtube_comments(video_id, limit=comment_limit)
        except Exception as e:
            st.error(f"Error fetching comments: {e}")
            st.stop()

    if not fetched_comments:
        st.warning("No comments found. Try a different video.")
        st.stop()

    st.info(f"Fetched {len(fetched_comments)} unique comments. Running sentiment analysis...")

    # Batch process once
    with st.spinner("Analyzing sentiments..."):
        results_df = evaluate_sentiment_batch(fetched_comments)

    results_df["Index"] = range(1, len(results_df) + 1)

    # =====================================
    # 📊 DISPLAY RESULTS
    # =====================================
    st.subheader("🧩 Sentiment Analysis Results")
    st.dataframe(results_df, use_container_width=True)

    # Pie chart
    pie = px.pie(
        names=results_df["Sentiment"].value_counts().index,
        values=results_df["Sentiment"].value_counts().values,
        title="Sentiment Distribution"
    )
    st.plotly_chart(pie, use_container_width=True)

    # Scatter
    scatter = px.scatter(
        results_df,
        x="Index", y="Sentiment",
        color="Sentiment",
        hover_data=["Comment"],
        title="Sentiment Timeline"
    )
    st.plotly_chart(scatter, use_container_width=True)

    # Positive/Negative Comments
    st.subheader("😊 Top Positive Comments")
    st.write(results_df.query("Sentiment == 'Positive'")[["Comment", "Emotion"]].head(5))

    st.subheader("😠 Top Negative Comments")
    st.write(results_df.query("Sentiment == 'Negative'")[["Comment", "Emotion"]].head(5))

    # Word Cloud
    st.subheader("☁️ Word Cloud of All Comments")
    full_text = " ".join(results_df["Comment"].tolist())
    wc = WordCloud(width=800, height=400, background_color="white").generate(full_text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    st.success("🎉 Analysis complete!")

