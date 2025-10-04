import re
import logging
from typing import Dict, List, Optional, Any
import nltk
from textblob import TextBlob
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

class IntentExtractor:
    def __init__(self):
        """Initialize the intent extraction system."""
        self.emotion_classifier = None
        self._load_models()

        # Define mood mappings
        self.mood_keywords = {
            'relaxed': ['calm', 'peaceful', 'relaxed', 'chill', 'zen', 'tranquil', 'serene'],
            'energetic': ['energetic', 'active', 'pumped', 'excited', 'lively', 'vigorous'],
            'happy': ['happy', 'joyful', 'cheerful', 'upbeat', 'positive', 'glad'],
            'creative': ['creative', 'artistic', 'imaginative', 'innovative', 'inspired'],
            'social': ['social', 'together', 'friends', 'people', 'group', 'community'],
            'focused': ['focused', 'concentrated', 'productive', 'work', 'study', 'learn'],
            'stressed': ['stressed', 'anxious', 'overwhelmed', 'tense', 'worried', 'pressure'],
            'adventurous': ['adventure', 'explore', 'outdoor', 'nature', 'hiking', 'travel']
        }

        # Time preference patterns
        self.time_patterns = {
            r'(\d+)\s*min': 'minutes',
            r'(\d+)\s*hour': 'hours',
            r'quick|short|brief': '15-30',
            r'medium|moderate': '30-60',
            r'long|extended': '60-120',
            r'all day|whole day': '120+'
        }

        # Activity type keywords
        self.activity_types = {
            'physical': ['exercise', 'workout', 'sport', 'run', 'gym', 'fitness', 'active'],
            'mental': ['read', 'study', 'learn', 'think', 'puzzle', 'brain', 'mind'],
            'creative': ['create', 'art', 'draw', 'paint', 'write', 'music', 'craft'],
            'social': ['friend', 'people', 'group', 'party', 'social', 'together'],
            'outdoor': ['outside', 'outdoor', 'nature', 'park', 'hiking', 'fresh air'],
            'indoor': ['inside', 'indoor', 'home', 'cozy', 'comfortable'],
            'wellness': ['relax', 'meditate', 'breathe', 'wellness', 'self-care', 'health']
        }

    def _load_models(self):
        """Load pre-trained models for emotion detection."""
        try:
            logger.info("Loading emotion classification model...")
            # Use a lightweight emotion classification model
            self.emotion_classifier = pipeline(
                "text-classification", 
                model="j-hartmann/emotion-english-distilroberta-base",
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("Emotion classifier loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load emotion classifier: {e}")
            self.emotion_classifier = None

    def extract_intent(self, text: str) -> Dict[str, Any]:
        """
        Extract intent and preferences from user text.

        Args:
            text: User input text

        Returns:
            Dictionary containing extracted intent information
        """
        text_lower = text.lower()

        intent = {
            'original_text': text,
            'mood': self._extract_mood(text_lower),
            'time_preference': self._extract_time_preference(text_lower),
            'activity_types': self._extract_activity_types(text_lower),
            'keywords': self._extract_keywords(text_lower),
            'sentiment': self._analyze_sentiment(text),
            'emotions': self._detect_emotions(text),
            'urgency': self._detect_urgency(text_lower)
        }

        return intent

    def _extract_mood(self, text: str) -> List[str]:
        """Extract mood from text."""
        moods = []

        for mood, keywords in self.mood_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    moods.append(mood)
                    break

        # If no specific mood found, try to infer from emotions
        if not moods and self.emotion_classifier:
            try:
                emotions = self.emotion_classifier(text[:512])  # Limit text length
                if emotions:
                    emotion_label = emotions[0]['label'].lower()
                    mood_mapping = {
                        'joy': 'happy',
                        'sadness': 'relaxed',
                        'anger': 'energetic',
                        'fear': 'stressed',
                        'surprise': 'excited',
                        'disgust': 'focused'
                    }
                    if emotion_label in mood_mapping:
                        moods.append(mood_mapping[emotion_label])
            except Exception as e:
                logger.error(f"Error in emotion detection: {e}")

        return moods if moods else ['relaxed']  # Default mood

    def _extract_time_preference(self, text: str) -> str:
        """Extract time preference from text."""
        # Look for specific time mentions
        for pattern, time_range in self.time_patterns.items():
            if re.search(pattern, text):
                if time_range in ['minutes', 'hours']:
                    # Extract the number
                    match = re.search(r'(\d+)', text)
                    if match:
                        num = int(match.group(1))
                        if 'min' in text:
                            return f"{num}-{num+15}"
                        elif 'hour' in text:
                            return f"{num*60}-{(num+1)*60}"
                else:
                    return time_range

        # Default time preference
        return '30-60'

    def _extract_activity_types(self, text: str) -> List[str]:
        """Extract activity types from text."""
        types = []

        for activity_type, keywords in self.activity_types.items():
            for keyword in keywords:
                if keyword in text:
                    types.append(activity_type)
                    break

        return types if types else ['wellness']  # Default type

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text."""
        # Simple keyword extraction using TextBlob
        blob = TextBlob(text)

        # Get noun phrases and filter
        keywords = []
        for phrase in blob.noun_phrases:
            if len(phrase) > 2:  # Ignore very short phrases
                keywords.append(phrase)

        # Add individual words that might be important
        words = blob.words
        important_words = [word for word in words if len(word) > 3 and word.isalpha()]
        keywords.extend(important_words[:5])  # Limit to top 5

        return list(set(keywords))  # Remove duplicates

    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of the text."""
        blob = TextBlob(text)

        return {
            'polarity': blob.sentiment.polarity,  # -1 to 1
            'subjectivity': blob.sentiment.subjectivity  # 0 to 1
        }

    def _detect_emotions(self, text: str) -> List[Dict[str, Any]]:
        """Detect emotions using the transformer model."""
        if not self.emotion_classifier:
            return []

        try:
            # Limit text length for the model
            truncated_text = text[:512]
            emotions = self.emotion_classifier(truncated_text)

            # Return top emotions with confidence scores
            return emotions[:3] if isinstance(emotions, list) else [emotions]

        except Exception as e:
            logger.error(f"Error in emotion detection: {e}")
            return []

    def _detect_urgency(self, text: str) -> str:
        """Detect urgency level from text."""
        urgent_keywords = ['urgent', 'asap', 'quickly', 'immediately', 'right now', 'need now']
        moderate_keywords = ['soon', 'today', 'this evening', 'later']

        for keyword in urgent_keywords:
            if keyword in text:
                return 'high'

        for keyword in moderate_keywords:
            if keyword in text:
                return 'medium'

        return 'low'

    def get_intent_summary(self, intent: Dict[str, Any]) -> str:
        """Generate a human-readable summary of the extracted intent."""
        mood_str = ', '.join(intent['mood']) if intent['mood'] else 'neutral'
        types_str = ', '.join(intent['activity_types']) if intent['activity_types'] else 'general'

        summary = f"Mood: {mood_str} | "
        summary += f"Time: {intent['time_preference']} minutes | "
        summary += f"Types: {types_str} | "
        summary += f"Urgency: {intent['urgency']}"

        return summary
