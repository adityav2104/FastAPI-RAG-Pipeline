from config import GOOGLE_API_KEY, GROQ_API_KEY
from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq

if not GOOGLE_API_KEY:
    raise ValueError("API not set")


gemini_llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash" , google_api_key = GOOGLE_API_KEY, temperature = 0.2)

groq_llm = ChatGroq(model="llama3-8b-8192", groq_api_key= GROQ_API_KEY, temperature=0.2)

