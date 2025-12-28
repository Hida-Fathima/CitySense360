import os
import sys

# Add project root to path so we can import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from src.inference.config import Config

def create_vector_db():
    print(f"🚀 Starting Knowledge Base Ingestion from: {Config.DATA_SOURCE_PATH}")
    
    # 1. Load PDFs
    if not os.path.exists(Config.DATA_SOURCE_PATH):
        print(f"❌ Error: Folder not found at {Config.DATA_SOURCE_PATH}")
        print("   -> Please create 'data/knowledge_base' and put your PDFs there.")
        return
        
    loader = PyPDFDirectoryLoader(str(Config.DATA_SOURCE_PATH))
    documents = loader.load()
    
    if not documents:
        print("⚠️ No PDFs found! Please add .pdf files to data/knowledge_base/")
        return
        
    print(f"📄 Loaded {len(documents)} pages.")

    # 2. Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE, 
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    docs = text_splitter.split_documents(documents)
    print(f"🧩 Split into {len(docs)} chunks.")

    # 3. Create Embeddings
    print(f"🧠 Loading Embeddings ({Config.EMBEDDING_MODEL_NAME})...")
    embeddings = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'} # Change to 'cuda' if you have GPU setup for torch
    )

    # 4. Save to FAISS
    print("💾 Building Vector Database...")
    db = FAISS.from_documents(docs, embeddings)
    
    os.makedirs(os.path.dirname(Config.VECTOR_STORE_PATH), exist_ok=True)
    db.save_local(str(Config.VECTOR_STORE_PATH))
    print(f"✅ Success! Knowledge Base saved to: {Config.VECTOR_STORE_PATH}")

if __name__ == "__main__":
    create_vector_db()