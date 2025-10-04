import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import logging
from typing import List, Dict, Any, Optional
import os


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActivityRecommender:
    def __init__(self, data_path: str = "data/activities.csv"):
        """Initialize the activity recommender."""
        self.data_path = data_path
        self.activities_df = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.scaler = StandardScaler()

        self._load_activities()
        self._prepare_features()

    def _load_activities(self):
        """Load activities dataset."""
        try:
            if os.path.exists(self.data_path):
                self.activities_df = pd.read_csv(self.data_path)
                logger.info(f"Loaded {len(self.activities_df)} activities from {self.data_path}")
            else:
                logger.warning(f"Data file {self.data_path} not found. Creating sample data.")
                self._create_sample_data()
        except Exception as e:
            logger.error(f"Error loading activities: {e}")
            self._create_sample_data()

    def _create_sample_data(self):
        """Create sample activities data if file doesn't exist."""
        sample_data = {
            'activity': [
                'Take a 10-minute walk in nature',
                'Practice deep breathing exercises',
                'Listen to calming music',
                'Do light stretching',
                'Read a book'
            ],
            'category': ['Wellness', 'Wellness', 'Wellness', 'Wellness', 'Learning'],
            'tags': [
                'nature, walking, outdoor, peaceful',
                'relaxation, breathing, calm, mindfulness',
                'music, relaxation, peaceful, audio',
                'flexibility, gentle, movement, body',
                'reading, quiet, learning, literature'
            ],
            'mood': ['calm', 'relaxed', 'peaceful', 'calm', 'curious'],
            'time_minutes': [10, 15, 30, 20, 45]
        }
        self.activities_df = pd.DataFrame(sample_data)
        logger.info("Created sample activities dataset")

    def _prepare_features(self):
        """Prepare features for recommendation."""
        if self.activities_df is None or self.activities_df.empty:
            logger.error("No activities data available")
            return

        # Combine text features
        self.activities_df['combined_features'] = (
            self.activities_df['activity'] + ' ' +
            self.activities_df['category'] + ' ' +
            self.activities_df['tags'] + ' ' +
            self.activities_df['mood']
        )

        # Create TF-IDF matrix
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.activities_df['combined_features']
        )
        logger.info("Features prepared for recommendation")

    def recommend_activities(self, intent: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Recommend activities based on user intent.

        Args:
            intent: Extracted intent from user input
            top_k: Number of recommendations to return

        Returns:
            List of recommended activities with scores
        """
        if self.activities_df is None or self.tfidf_matrix is None:
            logger.error("Recommender not properly initialized")
            return []

        try:
            # Create query vector from intent
            query_text = self._create_query_from_intent(intent)
            query_vector = self.tfidf_vectorizer.transform([query_text])

            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

            # Apply filters
            filtered_indices = self._apply_filters(intent)

            # Combine scores with filters
            final_scores = self._combine_scores(similarities, intent, filtered_indices)

            # Get top recommendations
            top_indices = np.argsort(final_scores)[::-1][:top_k]

            recommendations = []
            for idx in top_indices:
                if final_scores[idx] > 0:  # Only include positive scores
                    activity = self.activities_df.iloc[idx]
                    rec = {
                        'activity': activity['activity'],
                        'category': activity['category'],
                        'mood': activity['mood'],
                        'time_minutes': int(activity['time_minutes']),  # Convert here
                        'tags': activity['tags'],
                        'score': float(final_scores[idx]),  # Convert here
                        'similarity': float(similarities[idx])  # Convert here
                    }
                    recommendations.append(rec)

            logger.info(f"Generated {len(recommendations)} recommendations")
            return recommendations

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return self._get_fallback_recommendations(intent, top_k)

    def _create_query_from_intent(self, intent: Dict[str, Any]) -> str:
        """Create a query string from user intent."""
        query_parts = []

        # Add mood
        if intent.get('mood'):
            query_parts.extend(intent['mood'])

        # Add activity types
        if intent.get('activity_types'):
            query_parts.extend(intent['activity_types'])

        # Add keywords
        if intent.get('keywords'):
            query_parts.extend(intent['keywords'][:3])  # Limit keywords

        # Add sentiment-based terms
        sentiment = intent.get('sentiment', {})
        if sentiment.get('polarity', 0) > 0.1:
            query_parts.append('positive happy')
        elif sentiment.get('polarity', 0) < -0.1:
            query_parts.append('calm relaxing peaceful')

        return ' '.join(query_parts)

    def _apply_filters(self, intent: Dict[str, Any]) -> np.ndarray:
        """Apply filters based on user intent."""
        indices = np.ones(len(self.activities_df), dtype=bool)

        # Time preference filter
        time_pref = intent.get('time_preference', '30-60')
        if '-' in time_pref:
            try:
                min_time, max_time = map(int, time_pref.split('-'))
                time_mask = (
                    (self.activities_df['time_minutes'] >= min_time) &
                    (self.activities_df['time_minutes'] <= max_time)
                )
                indices &= time_mask
            except ValueError:
                pass  # Invalid time format, ignore filter

        # Mood filter (soft filter - boost rather than exclude)
        mood_filter = np.ones(len(self.activities_df))
        if intent.get('mood'):
            user_moods = intent['mood']
            for i, activity_mood in enumerate(self.activities_df['mood']):
                if activity_mood in user_moods:
                    mood_filter[i] = 1.5  # Boost matching moods

        return indices

    def _combine_scores(self, similarities: np.ndarray, intent: Dict[str, Any], 
                       filtered_indices: np.ndarray) -> np.ndarray:
        """Combine similarity scores with various factors."""
        scores = similarities.copy()

        # Apply filter mask
        scores = scores * filtered_indices

        # Boost based on mood match
        if intent.get('mood'):
            user_moods = intent['mood']
            for i, activity_mood in enumerate(self.activities_df['mood']):
                if activity_mood in user_moods:
                    scores[i] *= 1.3

        # Boost based on category match
        if intent.get('activity_types'):
            user_types = intent['activity_types']
            category_mapping = {
                'physical': ['Fitness'],
                'mental': ['Learning'],
                'creative': ['Creative'],
                'social': ['Social'],
                'wellness': ['Wellness'],
                'outdoor': ['Fitness', 'Wellness'],
                'indoor': ['Learning', 'Creative', 'Wellness']
            }

            for user_type in user_types:
                if user_type in category_mapping:
                    matching_categories = category_mapping[user_type]
                    for i, category in enumerate(self.activities_df['category']):
                        if category in matching_categories:
                            scores[i] *= 1.2

        # Add small random factor to break ties
        scores += np.random.random(len(scores)) * 0.01

        return scores

    def _get_fallback_recommendations(self, intent: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
        """Get fallback recommendations if main algorithm fails."""
        try:
            # Simple fallback based on mood
            mood = intent.get('mood', ['relaxed'])[0] if intent.get('mood') else 'relaxed'

            # Filter activities by mood
            matched_activities = self.activities_df[
                self.activities_df['mood'].str.contains(mood, case=False, na=False)
            ]

            if matched_activities.empty:
                matched_activities = self.activities_df.head(top_k)

            recommendations = []
            for _, activity in matched_activities.head(top_k).iterrows():
                rec = {
                    'activity': activity['activity'],
                    'category': activity['category'],
                    'mood': activity['mood'],
                    'time_minutes': int(activity['time_minutes']),
                    'tags': activity['tags'],
                    'score': 0.5,  # Default score
                    'similarity': 0.5
                }
                recommendations.append(rec)

            return recommendations

        except Exception as e:
            logger.error(f"Fallback recommendations failed: {e}")
            return []

    def get_activity_stats(self) -> Dict[str, Any]:
        """Get statistics about the activities dataset."""
        if self.activities_df is None:
            return {}

        stats = {
            'total_activities': len(self.activities_df),
            'categories': self.activities_df['category'].value_counts().to_dict(),
            'moods': self.activities_df['mood'].value_counts().to_dict(),
            'avg_time': float(self.activities_df['time_minutes'].mean()),
            'time_range': {
                'min': int(self.activities_df['time_minutes'].min()),
                'max': int(self.activities_df['time_minutes'].max())
            }
        }

        return stats

    def get_all_activities(self) -> List[Dict[str, Any]]:
        """Get all activities in the dataset."""
        if self.activities_df is None:
            return []

        activities = []
        for _, activity in self.activities_df.iterrows():
            activities.append({
                'activity': activity['activity'],
                'category': activity['category'],
                'mood': activity['mood'],
                'time_minutes': int(activity['time_minutes']),
                'tags': activity['tags']
            })

        return activities

    def add_activity(self, activity_data: Dict[str, Any]) -> bool:
        """Add a new activity to the dataset."""
        try:
            new_row = pd.DataFrame([activity_data])
            self.activities_df = pd.concat([self.activities_df, new_row], ignore_index=True)

            # Re-prepare features
            self._prepare_features()

            logger.info(f"Added new activity: {activity_data.get('activity', 'Unknown')}")
            return True

        except Exception as e:
            logger.error(f"Error adding activity: {e}")
            return False
