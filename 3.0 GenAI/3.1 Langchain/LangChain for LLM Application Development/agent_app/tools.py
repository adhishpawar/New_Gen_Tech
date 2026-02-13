from langchain_core.tools import tool
from rag_pipeline import build_rag_pipeline
import requests

# Load RAG pipeline
qa_chain = build_rag_pipeline("demo.txt")

@tool
def rag_tool(query: str) -> str:
    """Search the knowledge base (RAG) and return the most relevant answer."""
    result = qa_chain.invoke({"query": query})
    return result['result']

@tool
def calculator_tool(expression: str) -> str:
    """Solve a mathematical expression like '2 + 2 * 3' and return the result."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

@tool
def joke_tool(topic: str) -> str:
    """Return a joke about the given topic. Topics: 'python', 'math', or others."""
    try:
        jokes = {
            "python": "Why did the Python programmer go broke? Because he kept using eval() on his bank account.",
            "math": "Why was the equal sign so humble? Because it knew it wasn't less than or greater than anyone else."
        }
        return jokes.get(topic.lower(), "No joke found for that topic.")
    except Exception as e:
        return f"Error: {e}"
