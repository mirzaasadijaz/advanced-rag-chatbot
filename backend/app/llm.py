"""
Factory for Google Gemini chat models. Every node asks for its own LLM handle here
instead of importing ChatGoogleGenerativeAI directly, so tagging/temperature/streaming
stay consistent and easy to change in one place.
"""
import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from the .env file
load_dotenv()

def get_llm(
    temperature: float = 0.0,
    streaming: bool = False,
    tags: Optional[List[str]] = None,
) -> ChatGoogleGenerativeAI:
    
    # Fetch the API key directly from the environment
    google_api_key = os.getenv("GOOGLE_API_KEY")
    
    if not google_api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Please add it to your .env file. "
            "You can get a free key from https://aistudio.google.com/app/apikey"
        )
    
    # Optionally load the model name from .env, defaulting to 3.1-pro
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
        
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=google_api_key,
        temperature=temperature,
        streaming=streaming,
    )
    
    if tags:
        llm = llm.with_config(tags=tags)
        
    return llm