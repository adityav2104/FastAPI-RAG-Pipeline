import os
from config import GOOGLE_API_KEY
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama


if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set. Please check your .env file.")


gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)


llama_llm = ChatOllama(
    model="llama3",
    temperature=0.2
)
