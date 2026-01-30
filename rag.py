from langchain.chat_models import init_chat_model
from db import vector_store

llm = init_chat_model("google_genai:gemini-2.5-flash-lite")

def rag_query(query: str, k: int = 5):
    docs = vector_store.similarity_search(query, k=k)

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are an AI tutor.
Use ONLY the context below to answer.

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": [d.metadata for d in docs]
    }
