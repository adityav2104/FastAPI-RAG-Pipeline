import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import db_path, csv_path

def prepare_db():
    data = pd.read_csv(csv_path)
    sentence = [f"Question: {row['question']} Answer: {row['answer']}" for _, row in data.iterrows() ]
    embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
    db = FAISS.from_texts(sentence, embeddings)
    db.save_local(db_path)


if __name__ == "__main__":
    prepare_db()
