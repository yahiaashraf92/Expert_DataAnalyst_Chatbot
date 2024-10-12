import os
import getpass
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


def groq_llm_initializer():
    # Fetch the API key from environment variables
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment variables.")
    
    # Initialize ChatGroq with the correct API key
    return ChatGroq(api_key=GROQ_API_KEY, model="llama3-70b-8192")
