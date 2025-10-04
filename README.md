# 🎙️ Voice-to-Activity Recommendation System

A production-ready AI/ML system that converts speech to personalized activity recommendations using advanced NLP and machine learning techniques.

## ✨ Features

- **🎙️ Speech-to-Text**: Convert voice input using OpenAI Whisper or SpeechRecognition
- **🧠 Intent Extraction**: Extract mood, time preferences using HuggingFace Transformers
- **🎯 Smart Recommendations**: Content-based filtering with personalized suggestions
- **🚀 REST API**: FastAPI backend with comprehensive endpoints
- **💻 Interactive Frontend**: Beautiful Streamlit web application
- **🐳 Production Ready**: Docker containerization for easy deployment

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   Audio Input   │───▶│  Speech-to-Text  │───▶│   Intent Extraction │
│                 │    │   (Whisper)      │    │   (DistilBERT)      │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
                                                           │
┌─────────────────┐    ┌──────────────────┐              │
│ Recommendations │◀───│   ML Recommender │◀─────────────┘
│                 │    │ (Content-Based)  │
└─────────────────┘    └──────────────────┘
```

## 📊 Dataset

The system includes a curated dataset of 36+ activities with:
- **Activity**: Description of the activity
- **Category**: Type (Wellness, Entertainment, Outdoor, Creative, Social, etc.)
- **Tags**: Relevant keywords and descriptors
- **Mood**: Associated mood (relaxing, happy, creative, energetic, etc.)
- **Time**: Estimated duration in minutes

## 🚀 Quick Start (Beginner-Friendly)

### Prerequisites

- **Python 3.9+** (Download from [python.org](https://python.org))
- **Git** (Download from [git-scm.com](https://git-scm.com))

### Method 1: Local Installation (Recommended for beginners)

1. **Download the project:**
   ```bash
   # Extract the provided zip file to a folder like "voice-activity-recommender"
   # Open terminal/command prompt and navigate to the folder
   cd voice-activity-recommender
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   # Create virtual environment
   python -m venv venv

   # Activate it
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the backend API:**
   ```bash
   cd app
   python main.py
   ```

   You should see: `INFO: Uvicorn running on http://0.0.0.0:8000`

5. **Start the frontend (in a new terminal):**
   ```bash
   # Open a new terminal, navigate to project folder, and activate venv again
   cd voice-activity-recommender
   # Windows: venv\Scripts\activate
   # Mac/Linux: source venv/bin/activate

   streamlit run frontend/streamlit_app.py
   ```

   The web app will open automatically at `http://localhost:8501`

### Method 2: Docker (For advanced users)

1. **Build and run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

2. **Access the applications:**
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:8000

## 📁 Project Structure

```
voice-activity-recommender/
│
├── data/
│   └── activities.csv              # Activity dataset
│
├── models/
│   └── intent_model/              # Cached ML models
│
├── app/
│   ├── main.py                    # FastAPI application
│   ├── speech_to_text.py          # Speech-to-text processing
│   ├── nlp_intent.py             # Intent extraction
│   └── recommender.py            # Recommendation engine
│
├── frontend/
│   └── streamlit_app.py          # Streamlit web interface
│
├── requirements.txt              # Python dependencies
├── Dockerfile                   # Container configuration
├── docker-compose.yml          # Multi-container setup
└── README.md                   # This file
```

## 🎯 How to Use

### 1. **Open the Web Interface**
   - Go to http://localhost:8501 in your browser
   - You'll see a beautiful interface with multiple tabs

### 2. **Get Recommendations Tab**
   - Choose between **Text Input** or **Audio Upload**

   **Text Input Example:**
   ```
   "I feel stressed and need something calming for 20 minutes"
   ```

   **Audio Upload:**
   - Record yourself saying your preferences
   - Upload WAV, MP3, or other audio files
   - The system will transcribe and analyze your speech

### 3. **View Results**
   - Get personalized activity recommendations
   - See match scores and time requirements
   - View detailed intent analysis

### 4. **Explore Other Features**
   - **Analytics Tab**: View system statistics and charts
   - **Browse Activities**: See all available activities with filters
   - **About Tab**: Learn more about the system

## 💡 Example Inputs to Try

### Text Examples:
- "I'm feeling energetic and want something physical for about an hour"
- "I need to relax after work, something peaceful for 30 minutes"  
- "I want to be creative this evening"
- "Something fun and social with friends"
- "I'm stressed and need quick stress relief"

### Audio Examples:
- Record yourself saying any of the above
- Speak naturally and clearly
- Include your mood, time preference, and activity type

## 🔧 API Endpoints (For developers)

- `GET /` - API information
- `GET /health` - Health check
- `POST /recommend/text` - Text recommendations
- `POST /recommend/audio` - Audio recommendations  
- `GET /stats` - System statistics
- `GET /activities` - Browse activities

### Example API Usage:

```bash
# Text recommendation
curl -X POST "http://localhost:8000/recommend/text" \
     -H "Content-Type: application/json" \
     -d '{"text": "I feel stressed and need something relaxing"}'

# Audio recommendation
curl -X POST "http://localhost:8000/recommend/audio" \
     -F "audio_file=@your_audio.wav"
```

## 🛠️ Troubleshooting

### Common Issues:

1. **"ModuleNotFoundError" when starting:**
   ```bash
   # Make sure you're in the right directory and virtual environment is activated
   pip install -r requirements.txt
   ```

2. **"Connection refused" error in frontend:**
   - Make sure the backend is running (`python app/main.py`)
   - Check that it's running on port 8000

3. **Audio not working:**
   - Ensure audio file is clear and supported format (WAV, MP3, OGG)
   - Check file size (keep under 10MB)

4. **Slow performance:**
   - First startup is slow (downloading ML models)
   - Subsequent requests are much faster

5. **"Permission denied" errors:**
   - Make sure you have write permissions in the project folder
   - Try running as administrator if needed

### Getting Help:

1. **Check the logs** in the terminal where you started the backend
2. **Restart both backend and frontend** if things stop working
3. **Clear browser cache** if the frontend looks broken
4. **Check system requirements** (Python 3.9+, sufficient RAM)

## 📈 Performance Notes

- **First run**: Takes 1-2 minutes to download ML models
- **Subsequent runs**: Much faster (models are cached)
- **Audio processing**: Takes 5-15 seconds depending on file size
- **Text processing**: Nearly instant

## 🧪 Testing the System

### Test with these scenarios:

1. **Different moods**: happy, sad, stressed, excited, calm
2. **Time constraints**: 15 minutes, 1 hour, 2+ hours  
3. **Activity types**: creative, physical, social, solo
4. **Audio quality**: clear speech, background noise, different accents

### Expected Results:
- System should understand your intent accurately
- Recommendations should match your mood and preferences
- Scores should be reasonable (0.3-1.0 range)
- No crashes or errors in normal use

## 🔮 Future Enhancements

- **User Profiles**: Save preferences and history
- **Mobile App**: Native iOS/Android applications
- **Voice Recording**: Browser-based recording
- **More Languages**: Multi-language support
- **Integration**: Calendar and fitness app connectivity

## 📄 License

MIT License - Feel free to use and modify!

## 🙏 Acknowledgments

- OpenAI Whisper for speech recognition
- HuggingFace for NLP models
- FastAPI for the web framework
- Streamlit for rapid frontend development

---

**🎯 Ready to find your perfect activity? Start the system and let AI help you! 🚀**

## 📞 Support

If you encounter any issues:
1. Check this README for troubleshooting steps
2. Look at the terminal output for error messages
3. Ensure all prerequisites are installed correctly
4. Try restarting both backend and frontend services

**Happy recommending! 🎉**
