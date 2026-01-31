from fastapi import FastAPI, UploadFile, File
from ingest import ingest_pdf
from rag import rag_query
from schemas import IngestTextRequest
from db import vector_store
import shutil
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.chat_models import init_chat_model
from langchain.tools import tool


from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

import os
import uuid 



app = FastAPI(title="AI RAG Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # REQUIRED if using "*"
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest/pdf")
async def ingest_pdf_endpoint(file: UploadFile = File(...)):
    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    chunks = ingest_pdf(
        file_path,
        metadata={"source": "pdf", "filename": file.filename}
    )

    return {"status": "success", "chunks_added": chunks}

@app.post("/ingest/text")
def ingest_text(req: IngestTextRequest):
    vector_store.add_texts(
        texts=[req.text],
        metadatas=[req.metadata]
    )
    return {"status": "success"}

from pydantic import BaseModel
class QueryRequest(BaseModel):
    query: str
    k: int = 4

@app.post("/query")
async def query_docs(req: QueryRequest):
    retriever = vector_store.as_retriever(
        search_type="similarity",   # or "mmr"
        search_kwargs={"k": req.k}
    )

    docs = retriever.invoke(req.query)

    return {
        "query": req.query,
        "results": [
            {
                "content": d.page_content,
                "metadata": d.metadata
            }
            for d in docs
        ]
    }



# I gave up

from pydantic import BaseModel

class ChatRequest(BaseModel):
    msg: str
    k: int = 3


retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

from langchain_core.prompts import ChatPromptTemplate

template = """
You are a RAG assistant.

Answer the question using ONLY the information provided in the Context section.
Do NOT use any outside knowledge.
If the answer cannot be found in the context, respond exactly with:
"I don't know."

After answering, clearly explain the sources you used.

Rules for sources:
- List each source separately by the doc ID
- summarize what information was obtained from each source
- Mention what information came from each source
- Use only the provided metadata (source name, filename, page, etc.)
- Do not invent sources
- If multiple sources support the answer, explain how each contributes

Format your response exactly as follows:

Answer:
<your answer here>

Sources:
- Source 1: <source metadata> — <what this source supports>
- Source 2: <source metadata> — <what this source supports>


Context:
{context}

Question:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)



from langchain.chat_models import init_chat_model

model = init_chat_model("google_genai:gemini-2.0-flash")


from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

chain = (
    {
        "context": retriever
        | (lambda docs: "\n\n".join(d.page_content for d in docs)),
        "question": RunnablePassthrough(),
    }
    | prompt
    | model
    | StrOutputParser()
)


from fastapi import HTTPException

@app.post("/chat/")
async def quick_response(req: ChatRequest):
    try:
        result = await chain.ainvoke(req.msg)
        return {
            "query": req.msg,
            "response": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from pypdf import PdfReader

# ---- new -----
from fastapi import UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import uuid, shutil, os, json

@app.post("/check-syllabus/pdf")
async def check_syllabus_from_pdf(
    file: UploadFile = File(...),
    syllabus: str = Form(...)
):
    temp_path = None
    try:
        if not syllabus.strip():
            raise HTTPException(status_code=400, detail="Syllabus content is required")

        # 1. Save PDF temporarily
        temp_id = str(uuid.uuid4())
        temp_path = f"/tmp/{temp_id}_{file.filename}"

        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 2. Convert PDF → text
        reader = PdfReader(temp_path)
        pdf_text = ""

        for page in reader.pages:
            text = page.extract_text()
            if text:
                pdf_text += text + "\n\n"

        if not pdf_text.strip():
            raise HTTPException(status_code=400, detail="No readable text found in PDF")

        # 3. Prompt
        prompt = f"""
You are an academic content analyzer.

TASK:
1. Extract the main topics and subtopics from the PDF content.
2. Compare them with the syllabus provided.
3. Return:
   - Topics fully covered
   - Topics partially covered
   - Topics missing
   - Extra topics present in the PDF but not in the syllabus

PDF CONTENT:
{pdf_text}

SYLLABUS:
{syllabus}
IMPORTANT:
- Return ONLY valid JSON
- Do NOT include explanations
- Do NOT include markdown
- Do NOT include backticks
- Do NOT include any text outside JSON


OUTPUT FORMAT (STRICT JSON ONLY), use the below structure as template:
{{
  "covered_topics": [],
  "partially_covered_topics": [],
  "missing_topics": [],
  "extra_topics_in_pdf": [],
  "overall_coverage_percentage": 0
}}
"""

        # 4. Call LLM
        response = await model.ainvoke(prompt)

        # raw_text = response.content if hasattr(response, "content") else str(response)

# 5. Get raw LLM output
        raw_text = (
            response.content
            if hasattr(response, "content")
            else str(response)
        )

        # 6. Return output directly
        return {
            "status": "success",
            "filename": file.filename,
            "analysis": raw_text
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
