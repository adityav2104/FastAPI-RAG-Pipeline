from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_pipeline import app_graph

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_question(query: Query):
    try:
        result = dict(app_graph.invoke({"question": query.question}))
        return {
            "answer": result.get("final_answer", ""),
            "validation": result.get("validation", "")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
