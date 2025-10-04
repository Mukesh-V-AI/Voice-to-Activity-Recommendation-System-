import streamlit as st
import requests
import json
import io
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List, Any
import base64


# Configure Streamlit page
st.set_page_config(
    page_title=" Voice Activity Recommender",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# API Configuration (update if you use a different port or host)
API_BASE_URL = "http://localhost:8000"


# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .activity-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
    }
    .activity-title {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .activity-details {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)


class ActivityRecommenderUI:
    def __init__(self):
        self.api_base_url = API_BASE_URL

    def check_api_health(self) -> bool:
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def get_text_recommendations(self, text: str) -> Dict[str, Any]:
        try:
            response = requests.post(
                f"{self.api_base_url}/recommend/text",
                json={"text": text},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error getting recommendations: {str(e)}")
            return None

    def get_audio_recommendations(self, audio_bytes: bytes, filename: str) -> Dict[str, Any]:
        try:
            files = {"audio_file": (filename, audio_bytes, "audio/wav")}
            response = requests.post(
                f"{self.api_base_url}/recommend/audio",
                files=files,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        try:
            response = requests.get(f"{self.api_base_url}/stats", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error getting stats: {str(e)}")
            return None

    def get_all_activities(self) -> Dict[str, Any]:
        try:
            response = requests.get(f"{self.api_base_url}/activities", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error getting activities: {str(e)}")
            return None


def render_recommendation_card(rec: Dict[str, Any], index: int):
    with st.container():
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.markdown(f"### 🎯 {rec['activity']}")
            st.write(f"**Category:** {rec['category']}")
            st.write(f"**Tags:** {rec['tags']}")
        with col2:
            st.metric("Time", f"{rec['time_minutes']} min")
            st.write(f"**Mood:** {rec['mood']}")
        with col3:
            score_color = "green" if rec['score'] > 0.7 else "orange" if rec['score'] > 0.5 else "red"
            st.metric("Match Score", f"{rec['score']:.2f}")
            st.markdown(
                f"<div style='color: {score_color}'>Similarity: {rec['similarity']:.2f}</div>",
                unsafe_allow_html=True,
            )
        st.divider()


def render_intent_analysis(intent_data: Dict[str, Any]):
    st.subheader("🧠 Intent Analysis")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Detected Moods:**")
        moods = intent_data.get("mood", [])
        for mood in moods:
            st.badge(mood, type="secondary")
    with col2:
        st.write("**Activity Types:**")
        types = intent_data.get("activity_types", [])
        for activity_type in types:
            st.badge(activity_type, type="primary")
    with col3:
        st.write("**Time Preference:**")
        time_pref = intent_data.get("time_preference", "Not specified")
        st.info(f"{time_pref} minutes")

    sentiment = intent_data.get("sentiment", {})
    if sentiment:
        col1, col2 = st.columns(2)
        with col1:
            polarity = sentiment.get("polarity", 0)
            label = (
                "Positive"
                if polarity > 0.1
                else "Negative"
                if polarity < -0.1
                else "Neutral"
            )
            st.metric("Sentiment", label, f"{polarity:.2f}")
        with col2:
            subj = sentiment.get("subjectivity", 0)
            subj_label = "Subjective" if subj > 0.5 else "Objective"
            st.metric("Subjectivity", subj_label, f"{subj:.2f}")


def render_stats_dashboard(stats_data: Dict[str, Any]):
    if not stats_data:
        st.error("Could not load statistics")
        return

    dataset_stats = stats_data.get("dataset_stats", {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Activities", dataset_stats.get("total_activities", 0))
    with col2:
        avg_time = dataset_stats.get("avg_time", 0)
        st.metric("Avg Time", f"{avg_time:.0f} min")
    with col3:
        time_range = dataset_stats.get("time_range", {})
        st.metric("Time Range", f"{time_range.get('min', 0)}-{time_range.get('max', 0)} min")
    with col4:
        categories = dataset_stats.get("categories", {})
        st.metric("Categories", len(categories))

    col1, col2 = st.columns(2)

    with col1:
        if categories:
            st.subheader("Activities by Category")
            fig = px.pie(
                values=list(categories.values()),
                names=list(categories.keys()),
                title="Distribution of Activities by Category",
            )
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        moods = dataset_stats.get("moods", {})
        if moods:
            st.subheader("Activities by Mood")
            fig = px.bar(
                x=list(moods.keys()),
                y=list(moods.values()),
                title="Number of Activities by Mood",
            )
            st.plotly_chart(fig, use_container_width=True)


def main():
    ui = ActivityRecommenderUI()

    st.title("🎙️ Voice Activity Recommendation System")
    st.markdown("*AI-powered personalized activity suggestions from your voice or text input*")

    with st.sidebar:
        st.header("🔧 System Status")
        api_healthy = ui.check_api_health()
        if api_healthy:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Disconnected")
            st.warning("Please ensure the FastAPI backend is running on localhost:8000")

        st.divider()

        st.header("ℹ️ How it works")
        st.markdown(
            """
        1. **Input**: Speak or type your preferences  
        2. **Analysis**: AI extracts your mood and preferences  
        3. **Matching**: Smart algorithm finds perfect activities  
        4. **Results**: Get personalized recommendations
        """
        )

        st.divider()

        st.markdown("**Examples to try:**")
        st.code('"I feel stressed and need something relaxing"')
        st.code('"I want to exercise for 30 minutes"')
        st.code('"Something creative for this evening"')

    if not ui.check_api_health():
        st.error("🚫 Cannot connect to the API backend. Please start the FastAPI server first.")
        st.code("cd app && python main.py")
        return

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🎯 Get Recommendations", "📊 Analytics", "📋 Browse Activities", "ℹ️ About"]
    )

    with tab1:
        st.header("Get Your Personalized Activity Recommendations")

        input_method = st.radio(
            "Choose your input method:", ["💬 Text Input", "🎙️ Audio Upload"], horizontal=True
        )

        if input_method == "💬 Text Input":
            st.subheader("Text Input")

            st.write("**Quick Examples:**")
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("😰 Feeling Stressed"):
                    st.session_state.text_input = (
                        "I feel stressed and anxious, need something calming for 20 minutes"
                    )
            with col2:
                if st.button("⚡ Want Energy"):
                    st.session_state.text_input = (
                        "I feel energetic and want something physical for an hour"
                    )
            with col3:
                if st.button("🎨 Be Creative"):
                    st.session_state.text_input = "I want to do something creative and artistic this evening"

            user_text = st.text_area(
                "Describe what kind of activity you're looking for:",
                value=st.session_state.get("text_input", ""),
                placeholder="Example: I feel tired but want to do something relaxing for 30 minutes...",
                height=100,
            )

            if st.button("🔍 Get Recommendations", type="primary", disabled=not user_text.strip()):
                if user_text.strip():
                    with st.spinner("🤖 Analyzing your preferences and finding perfect activities..."):
                        result = ui.get_text_recommendations(user_text.strip())

                        if result:
                            st.success("✅ Found your perfect activities!")

                            st.info(f"**Understood:** {result['intent_summary']}")

                            recommendations = result["recommendations"]

                            if recommendations:
                                st.subheader(f"🎯 Top {len(recommendations)} Recommendations for You")

                                for i, rec in enumerate(recommendations, 1):
                                    with st.expander(f"#{i}: {rec['activity']}", expanded=i <= 2):
                                        render_recommendation_card(rec, i)

                                if st.checkbox("🔍 Show detailed analysis"):
                                    render_intent_analysis(result["processing_info"]["intent"])

                            else:
                                st.warning("No activities found matching your criteria. Try different keywords!")

        elif input_method == "🎙️ Audio Upload":
            st.subheader("Audio Upload")

            uploaded_file = st.file_uploader(
                "Upload your audio file:",
                type=["wav", "mp3", "ogg", "m4a"],
                help="Supported formats: WAV, MP3, OGG, M4A",
            )

            if uploaded_file is not None:
                st.audio(uploaded_file, format="audio/wav")

                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**File:** {uploaded_file.name}")
                with col2:
                    st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")

                if st.button("🎙️ Process Audio & Get Recommendations", type="primary"):
                    with st.spinner("🎧 Transcribing audio and analyzing preferences..."):
                        audio_bytes = uploaded_file.read()
                        result = ui.get_audio_recommendations(audio_bytes, uploaded_file.name)

                        if result:
                            st.success("✅ Audio processed successfully!")

                            transcribed_text = result["processing_info"].get("transcribed_text", "")
                            st.info(f"**You said:** \"{transcribed_text}\"")

                            st.info(f"**Understood:** {result['intent_summary']}")

                            recommendations = result["recommendations"]

                            if recommendations:
                                st.subheader(f"🎯 Top {len(recommendations)} Recommendations")

                                for i, rec in enumerate(recommendations, 1):
                                    with st.expander(f"#{i}: {rec['activity']}", expanded=i <= 2):
                                        render_recommendation_card(rec, i)

                                if st.checkbox("🔍 Show detailed analysis", key="audio_analysis"):
                                    render_intent_analysis(result["processing_info"]["intent"])

                            else:
                                st.warning("No activities found. Try describing your preferences more clearly!")

    with tab2:
        st.header("📊 System Analytics")

        if st.button("🔄 Refresh Data"):
            st.experimental_rerun()

        stats_data = ui.get_stats()
        if stats_data:
            render_stats_dashboard(stats_data)

    with tab3:
        st.header("📋 Browse All Activities")

        activities_data = ui.get_all_activities()
        if activities_data:
            activities = activities_data.get("activities", [])
            if activities:
                col1, col2, col3 = st.columns(3)

                with col1:
                    categories = list(set(activity["category"] for activity in activities))
                    selected_category = st.selectbox("Filter by Category", ["All"] + categories)
                with col2:
                    moods = list(set(activity["mood"] for activity in activities))
                    selected_mood = st.selectbox("Filter by Mood", ["All"] + moods)
                with col3:
                    max_time = st.slider("Maximum Time (minutes)", 0, 180, 180)

                filtered_activities = activities
                if selected_category != "All":
                    filtered_activities = [
                        a for a in filtered_activities if a["category"] == selected_category
                    ]
                if selected_mood != "All":
                    filtered_activities = [
                        a for a in filtered_activities if a["mood"] == selected_mood
                    ]
                filtered_activities = [
                    a for a in filtered_activities if a["time_minutes"] <= max_time
                ]

                st.write(f"**Showing {len(filtered_activities)} activities**")

                for i, activity in enumerate(filtered_activities):
                    with st.expander(f"{activity['activity']} ({activity['time_minutes']} min)"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Category:** {activity['category']}")
                            st.write(f"**Mood:** {activity['mood']}")
                        with col2:
                            st.write(f"**Time:** {activity['time_minutes']} minutes")
                            st.write(f"**Tags:** {activity['tags']}")

            else:
                st.error("No activities found in dataset")
        else:
            st.error("Could not load activities data")

    with tab4:
        st.header("ℹ️ About the System")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔧 Technical Components")
            st.markdown(
                """
            - **Speech-to-Text**: OpenAI Whisper + Google Speech Recognition  
            - **Intent Extraction**: HuggingFace Transformers (DistilRoBERTa)  
            - **Recommendations**: Content-based filtering with TF-IDF  
            - **Backend**: FastAPI with async processing  
            - **Frontend**: Streamlit with interactive UI  
            """
            )
            st.subheader("🎯 Features")
            st.markdown(
                """
            - Voice and text input support  
            - Real-time intent analysis  
            - Personalized recommendations  
            - Mood and preference matching  
            - Time-based filtering  
            - Interactive web interface  
            """
            )

        with col2:
            st.subheader("🚀 How to Use")
            st.markdown(
                """
            1. **Choose Input Method**: Voice recording or text  
            2. **Describe Preferences**: Tell us how you feel and what you want  
            3. **Get Recommendations**: AI analyzes and suggests activities  
            4. **Explore Options**: Browse detailed suggestions with scores  
            5. **Take Action**: Pick an activity and enjoy!  
            """
            )
            st.subheader("💡 Tips for Best Results")
            st.markdown(
                """
            - Be specific about your mood and energy level  
            - Mention time constraints if important  
            - Include activity preferences (indoor/outdoor, social/solo)  
            - Speak clearly for audio input  
            - Try different phrasings if results aren't perfect  
            """
            )

        st.divider()
        st.markdown("**Built with ❤️ using FastAPI, Streamlit, and modern AI/ML technologies**")


if __name__ == "__main__":
    main()
