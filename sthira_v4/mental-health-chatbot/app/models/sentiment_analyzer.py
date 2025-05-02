from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import numpy as np

class SentimentAnalyzer:
    def __init__(self):
        """
        Initialize sentiment analysis tools.
        Download necessary NLTK resources.
        """
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except Exception as e:
            print(f"NLTK download warning: {e}")
        
        self.vader_analyzer = SentimentIntensityAnalyzer()
    
    def analyze_sentiment(self, text: str) -> dict:
        """
        Perform multi-dimensional sentiment analysis.
        
        :param text: Input text for sentiment analysis
        :return: Comprehensive sentiment analysis dictionary
        """
        # TextBlob sentiment (polarity and subjectivity)
        blob_sentiment = TextBlob(text)
        
        # VADER sentiment
        vader_sentiment = self.vader_analyzer.polarity_scores(text)
        
        # Emotional tone detection
        emotional_tone = self._detect_emotional_tone(text)
        
        return {
            'polarity': blob_sentiment.sentiment.polarity,
            'subjectivity': blob_sentiment.sentiment.subjectivity,
            'vader_sentiment': vader_sentiment,
            'emotional_tone': emotional_tone,
            'risk_level': self._assess_mental_health_risk(vader_sentiment)
        }
    
    def _detect_emotional_tone(self, text: str) -> str:
        """
        Detect underlying emotional tone.
        
        :param text: Input text
        :return: Emotional tone description
        """
        emotional_keywords = {
            'anxiety': ['worry', 'nervous', 'panic', 'afraid', 'stress'],
            'depression': ['sad', 'hopeless', 'lonely', 'worthless', 'tired'],
            'anger': ['frustrated', 'mad', 'furious', 'irritated'],
            'neutral': ['okay', 'fine', 'alright']
        }
        
        text_lower = text.lower()
        
        for tone, keywords in emotional_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return tone
        
        return 'neutral'
    
    def _assess_mental_health_risk(self, vader_sentiment: dict) -> str:
        """
        Assess potential mental health risk based on sentiment.
        
        :param vader_sentiment: VADER sentiment scores
        :return: Risk level
        """
        compound_score = vader_sentiment['compound']
        
        if compound_score <= -0.5:
            return 'high_risk'
        elif -0.5 < compound_score <= -0.2:
            return 'moderate_risk'
        elif -0.2 < compound_score < 0.2:
            return 'low_risk'
        else:
            return 'positive'