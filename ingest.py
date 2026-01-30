from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from db import vector_store

def ingest_pdf(file_path: str, metadata: dict):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    splits = splitter.split_documents(docs)

    for doc in splits:
        doc.metadata.update(metadata)

    vector_store.add_documents(splits)
    return len(splits)


