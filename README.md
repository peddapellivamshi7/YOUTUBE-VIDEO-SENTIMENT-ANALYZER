# YOUTUBE-VIDEO-SENTIMENT-ANALYZER
Streamlit app that analyzes YouTube comments using Hugging Face’s RoBERTa model via the YouTube Data API. Fetch up to 500 comments, auto-refresh in real time, and visualize sentiment trends with pie charts, scatter plots, and word clouds. Export results to CSV for deeper audience insight.

🌐 **Try the live app:** [YouTube Video Sentiment Analyzer](https://youtube-video-sentiment-analyzer.streamlit.app/)

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


🧾 Usage

Enter a YouTube video URL or ID

Set refresh interval and comment limit

Click Start Sentiment Analysis

Explore:

Sentiment distribution (Pie Chart)

Comment timeline (Scatter Plot)

Word cloud summary

Most positive and negative comments

💡 Future Improvements

🌍 Add multilingual comment translation using deep-translator

❤️ Include fine-grained emotion detection model

💬 Support for nested comment replies

🗄️ Database integration for long-term analytics

🧑‍💻 Author

Peddapelli Vamshi – GitHub(https://github.com/peddapellivamshi7)

If you like this project, please ⭐ the repo and share your feedback!

🪪 License

This project is licensed under the MIT License – you are free to use and modify it.

🏷️ Tags

streamlit • nlp • huggingface • youtube-api • sentiment-analysis • python • data-visualization
