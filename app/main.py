from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tempfile
import os
import logging
from typing import List, Dict, Any, Optional
import uvicorn

from speech_to_text import SpeechToTextProcessor
from nlp_intent import IntentExtractor
from recommender import ActivityRecommender

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Voice Activity Recommendation System",
    description="AI-powered system that converts speech to personalized activity recommendations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
speech_processor = SpeechToTextProcessor(use_whisper=True)
intent_extractor = IntentExtractor()
recommender = ActivityRecommender()

# Pydantic models
class TextRequest(BaseModel):
    text: str

class RecommendationResponse(BaseModel):
    recommendations: List[Dict[str, Any]]
    intent_summary: str
    processing_info: Dict[str, Any]

class ActivityRequest(BaseModel):
    activity: str
    category: str
    tags: str
    mood: str
    time_minutes: int

@app.get("/")
async def root():
    """API information and available endpoints."""
    return {
        "message": "Voice Activity Recommendation System",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/recommend/text": "Get recommendations from text input",
            "/recommend/audio": "Get recommendations from audio file",
            "/stats": "System statistics",
            "/activities": "Browse all activities"
        },
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "components": {
            "speech_processor": "loaded" if speech_processor else "error",
            "intent_extractor": "loaded" if intent_extractor else "error",
            "recommender": "loaded" if recommender else "error"
        }
    }

@app.post("/recommend/text", response_model=RecommendationResponse)
async def recommend_from_text(request: TextRequest):
    """Get activity recommendations from text input."""
    try:
        logger.info(f"Processing text: {request.text}")

        # Extract intent from text
        intent = intent_extractor.extract_intent(request.text)
        intent_summary = intent_extractor.get_intent_summary(intent)

        # Get recommendations
        recommendations = recommender.recommend_activities(intent, top_k=5)

        processing_info = {
            "input_type": "text",
            "intent": intent,
            "recommendations_count": len(recommendations)
        }

        return RecommendationResponse(
            recommendations=recommendations,
            intent_summary=intent_summary,
            processing_info=processing_info
        )

    except Exception as e:
        logger.error(f"Error processing text recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend/audio", response_model=RecommendationResponse)
async def recommend_from_audio(audio_file: UploadFile = File(...)):
    """Get activity recommendations from audio file."""
    try:
        logger.info(f"Processing audio file: {audio_file.filename}")

        # Validate file type
        allowed_types = ['audio/wav', 'audio/mpeg', 'audio/mp4', 'audio/ogg', 'audio/webm']
        if audio_file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported audio format. Allowed: {allowed_types}"
            )

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            # Transcribe audio to text
            transcribed_text = speech_processor.transcribe_audio(temp_file_path)

            if not transcribed_text:
                raise HTTPException(
                    status_code=400, 
                    detail="Could not transcribe audio. Please ensure audio is clear."
                )

            logger.info(f"Transcribed text: {transcribed_text}")

            # Extract intent from transcribed text
            intent = intent_extractor.extract_intent(transcribed_text)
            intent_summary = intent_extractor.get_intent_summary(intent)

            # Get recommendations
            recommendations = recommender.recommend_activities(intent, top_k=5)

            processing_info = {
                "input_type": "audio",
                "transcribed_text": transcribed_text,
                "intent": intent,
                "recommendations_count": len(recommendations)
            }

            return RecommendationResponse(
                recommendations=recommendations,
                intent_summary=intent_summary,
                processing_info=processing_info
            )

        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing audio recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    try:
        stats = recommender.get_activity_stats()

        system_stats = {
            "dataset_stats": stats,
            "system_info": {
                "speech_processor": "Whisper + SpeechRecognition",
                "intent_extractor": "DistilRoBERTa + TextBlob",
                "recommender": "Content-based filtering with TF-IDF"
            }
        }

        return system_stats

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/activities")
async def get_activities():
    """Get all available activities."""
    try:
        activities = recommender.get_all_activities()
        return {
            "activities": activities,
            "total_count": len(activities)
        }

    except Exception as e:
        logger.error(f"Error getting activities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/activities")
async def add_activity(request: ActivityRequest):
    """Add a new activity to the dataset."""
    try:
        activity_data = {
            "activity": request.activity,
            "category": request.category,
            "tags": request.tags,
            "mood": request.mood,
            "time_minutes": request.time_minutes
        }

        success = recommender.add_activity(activity_data)

        if success:
            return {"message": "Activity added successfully", "activity": activity_data}
        else:
            raise HTTPException(status_code=500, detail="Failed to add activity")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding activity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
