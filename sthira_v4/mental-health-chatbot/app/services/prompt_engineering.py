import json
import random
from typing import List, Dict

class PromptEngineer:
    def __init__(self, prompts_file: str = 'data/mental_health_prompts.json'):
        """
        Initialize prompt engineering with specialized mental health prompts.
        
        :param prompts_file: Path to prompts JSON file
        """
        with open(prompts_file, 'r') as f:
            self.prompts = json.load(f)
    
    def construct_therapeutic_prompt(self, user_message: str, context: List[Dict], sentiment: dict) -> str:
        """
        Construct a context-aware, empathetic therapeutic prompt.
        
        :param user_message: Current user message
        :param context: Conversation history
        :param sentiment: Sentiment analysis results
        :return: Engineered therapeutic prompt
        """
        # Select appropriate prompt based on emotional tone
        emotional_tone = sentiment['emotional_tone']
        risk_level = sentiment['risk_level']
        
        # Base therapeutic framework
        base_prompts = self.prompts.get('base_framework', [])
        tone_specific_prompts = self.prompts.get(emotional_tone, [])
        
        # Combine prompts with contextual awareness
        context_prompt = self._generate_context_aware_prompt(context)
        
        # Risk-level specific guidance
        risk_guidance = self._get_risk_level_guidance(risk_level)
        
        # Construct final prompt
        prompt_components = [
            random.choice(base_prompts),
            context_prompt,
            random.choice(tone_specific_prompts),
            risk_guidance,
            f"User's current message: {user_message}"
        ]
        
        return " ".join(prompt_components)
    
    def _generate_context_aware_prompt(self, context: List[Dict]) -> str:
        """
        Generate context-aware therapeutic prompt.
        
        :param context: Conversation history
        :return: Context-aware prompt segment
        """
        if not context:
            return "Begin our conversation with empathy and active listening."
        
        # Analyze recent conversation sentiment
        recent_sentiments = [msg['sentiment'] for msg in context[-3:]]
        avg_sentiment = sum(recent_sentiments) / len(recent_sentiments)
        
        context_hints = {
            'positive': "Build upon the user's positive reflections.",
            'negative': "Provide compassionate support and validate feelings.",
            'neutral': "Gently explore underlying emotions."
        }
        
        sentiment_category = (
            'positive' if avg_sentiment > 0.2 else
            'negative' if avg_sentiment < -0.2 else
            'neutral'
        )
        
        return context_hints[sentiment_category]
    
    def _get_risk_level_guidance(self, risk_level: str) -> str:
        """
        Provide risk-level specific therapeutic guidance.
        
        :param risk_level: Mental health risk level
        :return: Risk-specific guidance
        """
        risk_guidance = {
            'high_risk': "Exercise extreme caution. Prioritize safety and professional intervention.",
            'moderate_risk': "Provide supportive, non-judgmental guidance. Suggest professional resources.",
            'low_risk': "Offer gentle, constructive emotional support.",
            'positive': "Reinforce positive emotional states and coping strategies."
        }
        
        return risk_guidance.get(risk_level, "Provide compassionate support.")