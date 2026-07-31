from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


def create_vector_store(chunks):
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )
    return FAISS.from_documents(chunks, embeddings)