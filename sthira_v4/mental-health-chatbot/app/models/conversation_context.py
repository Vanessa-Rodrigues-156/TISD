import uuid
from typing import List, Dict, Optional
from datetime import datetime, timedelta

class ConversationContext:
    def __init__(self, max_history_length: int = 10, context_timeout: int = 3600):
        """
        Manage conversation context with history and timeout.
        
        :param max_history_length: Maximum number of messages to retain
        :param context_timeout: Timeout for conversation context in seconds
        """
        self.conversations: Dict[str, Dict] = {}
    
    def create_conversation(self) -> str:
        """
        Create a new conversation session.
        
        :return: Unique conversation ID
        """
        conversation_id = str(uuid.uuid4())
        self.conversations[conversation_id] = {
            'messages': [],
            'created_at': datetime.now(),
            'sentiment_history': [],
            'mental_health_flags': []
        }
        return conversation_id
    
    def add_message(self, conversation_id: str, role: str, message: str, sentiment: float):
        """
        Add a message to conversation history.
        
        :param conversation_id: Unique identifier for the conversation
        :param role: 'user' or 'assistant'
        :param message: Message content
        :param sentiment: Sentiment score
        """
        if conversation_id not in self.conversations:
            raise ValueError("Invalid conversation ID")
        
        conversation = self.conversations[conversation_id]
        
        # Trim history if exceeding max length
        if len(conversation['messages']) >= 10:
            conversation['messages'].pop(0)
        
        conversation['messages'].append({
            'role': role,
            'content': message,
            'timestamp': datetime.now(),
            'sentiment': sentiment
        })
        
        # Track sentiment history
        conversation['sentiment_history'].append(sentiment)
    
    def get_conversation_context(self, conversation_id: str) -> Optional[List[Dict]]:
        """
        Retrieve conversation context.
        
        :param conversation_id: Unique identifier for the conversation
        :return: List of message contexts or None
        """
        if conversation_id not in self.conversations:
            return None
        
        conversation = self.conversations[conversation_id]
        
        # Check conversation timeout
        if datetime.now() - conversation['created_at'] > timedelta(seconds=3600):
            del self.conversations[conversation_id]
            return None
        
        return conversation['messages']
    
    def flag_mental_health_concern(self, conversation_id: str, concern: str):
        """
        Flag potential mental health concerns.
        
        :param conversation_id: Unique identifier for the conversation
        :param concern: Description of the mental health concern
        """
        if conversation_id in self.conversations:
            self.conversations[conversation_id]['mental_health_flags'].append({
                'concern': concern,
                'timestamp': datetime.now()
            })