from typing import TypedDict
from langgraph.graph import StateGraph, END
from llms import gemini_llm, llama_llm
from retriever import get_retriever
import json

class AppState(TypedDict):
    question: str
    context: str
    draft_answer: str
    final_answer: str
    validation: str

def retrieve_context(state: AppState):
    retriever = get_retriever()
    docs = retriever.invoke(state["question"])
    state["context"] = "\n".join(doc.page_content for doc in docs)
    return state

def generate_and_improve(state: AppState):
    gemini_prompt = f"Use this context to answer:\n{state['context']}\n\nQuestion: {state['question']}"
    response = gemini_llm.invoke(gemini_prompt)
    gemini_response = getattr(response, "content", str(response))
    state["draft_answer"] = gemini_response.strip()

   
    llama_prompt = f"""
    You are a helpful AI that improves answers for clarity and accuracy.
    Given this answer: {state['draft_answer']}

    1. Provide the improved answer.
    2. Briefly explain what you improved or validated.

    Return ONLY valid JSON in this exact format:
    {{
        "improved": "<your improved answer here>",
        "validation": "<your validation here>"
    }}
    """
    response2 = llama_llm.invoke(llama_prompt)
    llama_response = getattr(response2, "content", str(response2)).strip()

    try:
        parsed = json.loads(llama_response)
        improved = parsed.get("improved", "").strip()
        validation = parsed.get("validation", "").strip()

        if not improved:
            raise ValueError("Empty improved answer")

    except (json.JSONDecodeError, ValueError) as e:
        improved = state["draft_answer"]
        validation = f"Format Guardrail: Invalid JSON from LLaMA ({e}). Returned Gemini original answer."

 
    context_lower = state["context"].lower()
    overlap = sum(1 for word in improved.split() if word.lower() in context_lower)
    if overlap < len(improved.split()) * 0.2:  
        validation += " | Factuality Guardrail: Answer may not be grounded in retrieved context."
        improved = state["draft_answer"]

    state["final_answer"] = improved
    state["validation"] = validation
    return state

graph = StateGraph(AppState)
graph.add_node("retrieve", retrieve_context)
graph.add_node("generate_improve", generate_and_improve)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate_improve")
graph.add_edge("generate_improve", END)

app_graph = graph.compile()


