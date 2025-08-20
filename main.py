# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from rag_pipeline import app_graph

# app = FastAPI()

# class Query(BaseModel):
#     question: str

# @app.post("/ask")
# def ask_question(query: Query):
#     try:
#         result = dict(app_graph.invoke({"question": query.question}))
#         return {
#             "answer": result.get("final_answer", ""),
#             "validation": result.get("validation", "")
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from rag_pipeline import app_graph
import traceback

app = FastAPI()

class Query(BaseModel):
    question: str

# Health check endpoint
@app.get("/ping")
def ping():
    return {"status": "ok"}

# Global error logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal error: {str(e)}"},
        )

# Main RAG endpoint
@app.post("/ask")
# def ask_question(query: Query):
#     try:
#         result = dict(app_graph.invoke({"question": query.question}))
#         return {
#             "answer": result.get("final_answer", ""),
#             "validation": result.get("validation", "")
#         }
#     except Exception as e:
#         print("ASK ERROR:", e)
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=str(e))

def ask_question(query: Query):
    return {"answer": f"Echo: {query.question}", "validation": "dummy"}