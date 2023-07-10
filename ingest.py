"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle
import vertexai

from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import VertexAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

PROJECT_ID = "cloud-llm-preview4"
REGION = "us-central1"

def ingest_docs():
    """Initialize Vertex AI"""
    """gcloud config set project $PROJECT_ID"""
    """gcloud auth application-default login"""
    vertexai.init(project=PROJECT_ID, location=REGION)

    """Get documents from web pages."""
    loader = UnstructuredFileLoader("./source-docs/raw-xpress-cash.html")
    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(raw_documents)
    embeddings = VertexAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)


if __name__ == "__main__":
    ingest_docs()
