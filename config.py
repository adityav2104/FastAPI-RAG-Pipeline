import os
from dotenv import load_dotenv


db_path = "faiss_db"
csv_path = "data/cars.csv"
json_path = "qa_Automotive.json"
csv2_path = "data/cars2.csv"


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")