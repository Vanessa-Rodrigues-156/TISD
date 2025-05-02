from fastapi import FastAPI, Request, HTTPException # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from pydantic import BaseModel # type: ignore

from app.models.conversation_context import ConversationContext
from app.models.sentiment_analyzer import SentimentAnalyzer
from app.services.prompt_engineering import PromptEngineer
from app.services.rate_limiter import DistributedRateLimiter

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch # type: ignore
import os
from dotenv import load_dotenv # type: ignore



load_dotenv()  


class ChatRequest(BaseModel):
    message: str
    conversation_id: str = None

class MentalHealthChatbot:
    def __init__(self):
        # Initialize core components
        self.conversation_context = ConversationContext()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.prompt_engineer = PromptEngineer()
        self.rate_limiter = DistributedRateLimiter()
        
        # Define your Hugging Face token
        your_token = os.getenv("HUGGINGFACE_TOKEN")
        
        # Load therapeutic language model with authentication token
        self.tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.1", 
            use_auth_token=your_token
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.1",
            torch_dtype=torch.float16,
            device_map="auto",
            use_auth_token=your_token
        )
    
    def generate_response(self, message: str, context: list):
        """
        Generate empathetic therapeutic response.
        
        :param message: User's message
        :param context: Conversation context
        :return: AI-generated response
        """
        # Analyze sentiment
        sentiment = self.sentiment_analyzer.analyze_sentiment(message)
        
        # Engineer therapeutic prompt
        therapeutic_prompt = self.prompt_engineer.construct_therapeutic_prompt(
            message, context, sentiment
        )
        
        # Generate response using model
        inputs = self.tokenizer(therapeutic_prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_length=200)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response, sentiment

# FastAPI Application
app = FastAPI(
    title="Mental Health Companion",
    description="AI-powered therapeutic chatbot with advanced sentiment analysis"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chatbot = MentalHealthChatbot()

@app.post("/chat")
async def chat_endpoint(request: Request, chat_request: ChatRequest):
    """
    Primary chat endpoint for mental health interactions.
    
    :param request: FastAPI request object
    :param chat_request: Chat request payload
    :return: Therapeutic response with sentiment analysis
    """
    try:
        # Apply rate limiting
        chatbot.rate_limiter.limit_request(request)
        
        # Determine conversation ID
        conversation_id = (
            chat_request.conversation_id or 
            chatbot.conversation_context.create_conversation()
        )
        
        # Retrieve conversation context
        context = chatbot.conversation_context.get_conversation_context(conversation_id)
        
        # Generate response
        response, sentiment = chatbot.generate_response(
            chat_request.message, 
            context or []
        )
        
        # Add messages to conversation context
        chatbot.conversation_context.add_message(
            conversation_id, 
            'user', 
            chat_request.message, 
            sentiment['polarity']
        )
        chatbot.conversation_context.add_message(
            conversation_id, 
            'assistant', 
            response, 
            sentiment['polarity']
        )
        
        # Check for potential mental health concerns
        if sentiment['risk_level'] in ['high_risk', 'moderate_risk']:
            chatbot.conversation_context.flag_mental_health_concern(
                conversation_id, 
                f"Risk Level: {sentiment['risk_level']}"
            )
        
        return {
            "conversation_id": conversation_id,
            "message": response,
            "sentiment": sentiment
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
