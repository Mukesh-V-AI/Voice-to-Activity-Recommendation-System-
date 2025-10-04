from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Activity Recommender - Debug Version")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

@app.get("/")
async def root():
    return {
        "message": "Voice Activity Recommendation System - Debug Mode",
        "status": "running",
        "version": "debug-1.0"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "mode": "debug"}

@app.post("/recommend/text")
async def recommend_from_text(request: TextRequest):
    try:
        logger.info(f"Processing text: {request.text}")

        mock_recommendations = [
            {
                "activity": "Take a 10-minute walk in nature",
                "category": "Wellness",
                "mood": "calm",
                "time_minutes": 10,
                "tags": "nature, walking, outdoor, peaceful",
                "score": 0.95,
                "similarity": 0.87
            },
            {
                "activity": "Practice deep breathing exercises",
                "category": "Wellness",
                "mood": "relaxed",
                "time_minutes": 15,
                "tags": "relaxation, breathing, mindfulness",
                "score": 0.88,
                "similarity": 0.79
            }
        ]

        intent_summary = f"Understood: Looking for activities based on '{request.text}'"

        processing_info = {
            "input_type": "text",
            "intent": {
                "mood": ["relaxed"],
                "time_preference": "30-60",
                "activity_types": ["wellness"],
                "keywords": request.text.split()[:5]
            },
            "recommendations_count": len(mock_recommendations)
        }

        return {
            "recommendations": mock_recommendations,
            "intent_summary": intent_summary,
            "processing_info": processing_info
        }

    except Exception as e:
        logger.error(f"Error in simplified recommendation: {e}")
        raise HTTPException(status_code=500, detail=f"Debug error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main_debug:app", host="0.0.0.0", port=8001, reload=True, log_level="info")
