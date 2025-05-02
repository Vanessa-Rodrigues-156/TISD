from pydantic import BaseModel, Field
from typing import Optional, Dict

class ChatRequest(BaseModel):
    """
    Schema for incoming chat requests.
    """
    message: str = Field(..., min_length=1, max_length=1000)
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    """
    Schema for chat responses.
    """
    conversation_id: str
    message: str
    sentiment: Dict[str, float]
    risk_level: Optional[str] = None

class ErrorResponse(BaseModel):
    """
    Schema for error responses.
    """
    error_code: str
    message: str