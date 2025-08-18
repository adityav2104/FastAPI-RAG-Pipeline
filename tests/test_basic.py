from fastapi.testclient import TestClient
from unittest.mock import patch

from main import app

client = TestClient(app)


# ---- Helper mock responses ----
def mock_app_graph_invoke(input_data):
    # Simulate what app_graph.invoke would return
    question = input_data.get("question", "")
    return {
        "question": question,
        "context": "mock context",
        "draft_answer": "mock draft answer",
        "final_answer": f"Mock final answer for: {question}",
        "validation": "Mock validation successful"
    }


# ---- Tests ----
def test_ask_question_success():
    """Test /ask with a valid question"""
    with patch("rag_pipeline.app_graph.invoke", side_effect=mock_app_graph_invoke):
        response = client.post("/ask", json={"question": "What is ABS in cars?"})
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "validation" in data
        assert "ABS" in data["answer"] or "Mock" in data["answer"]


def test_ask_question_missing_field():
    """Test /ask with missing 'question' field"""
    response = client.post("/ask", json={})
    assert response.status_code == 422  # Unprocessable Entity (validation error)


def test_ask_question_internal_error():
    """Test /ask when pipeline throws exception"""
    with patch("rag_pipeline.app_graph.invoke", side_effect=Exception("Mock failure")):
        response = client.post("/ask", json={"question": "Any car?"})
        assert response.status_code == 500
        data = response.json()
        assert data["detail"] == "Mock failure"
