# This file can be left empty or used for package-level imports
from fastapi import FastAPI

def create_app() -> FastAPI:
    """
    Application factory function.
    """
    app = FastAPI(
        title="Mental Health Companion",
        description="AI-powered therapeutic chatbot with advanced sentiment analysis"
    )
    return app