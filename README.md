# YOUTUBE-VIDEO-SENTIMENT-ANALYZER
Streamlit app that analyzes YouTube comments using Hugging Face’s RoBERTa model via the YouTube Data API. Fetch up to 500 comments, auto-refresh in real time, and visualize sentiment trends with pie charts, scatter plots, and word clouds. Export results to CSV for deeper audience insight.

---

## 🔍 Features
- 📥 Fetch up to **500 comments** from any YouTube video  
- 🤖 Analyze sentiments using **RoBERTa** (state-of-the-art NLP)  
- 🔄 Real-time **auto-refresh** to track new comments  
- 📊 Visual insights with **Pie Charts**, **Timeline Plots**, and **Word Clouds**  
- 💾 Export analysis results as **CSV** for deeper study  

---

## 🧠 Tech Stack
- **Python 3.10+**  
- **Streamlit** – for interactive web UI  
- **Transformers (Hugging Face)** – for sentiment analysis  
- **Google API Client** – for YouTube Data API access  
- **Plotly**, **Matplotlib**, **WordCloud** – for data visualization  

---

## ⚙️ Setup Instructions

```bash
# 1️⃣ Clone the Repository
git clone https://github.com/peddapellivamshi7/YOUTUBE-VIDEO-SENTIMENT-ANALYZER.git
cd YOUTUBE-VIDEO-SENTIMENT-ANALYZER

# 2️⃣ Install Dependencies
pip install -r requirements.txt

# If you don’t have a requirements.txt, create one with the following content:
# --------------------------------------
# streamlit
# pandas
# google-api-python-client
# transformers
# torch
# wordcloud
# matplotlib
# plotly
# --------------------------------------

# 3️⃣ Add Your YouTube API Key
# Open app.py and replace the placeholder with your actual API key:
# API_KEY = "YOUR_YOUTUBE_API_KEY"
# (You can get your key from Google Cloud Console: https://console.cloud.google.com/)

# 4️⃣ Run the App
streamlit run app.py

# After running, open the URL shown in your terminal (usually http://localhost:8501)

