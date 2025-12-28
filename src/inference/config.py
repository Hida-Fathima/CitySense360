import os
from pathlib import Path

class Config:
    # --- File Paths ---
    # We go up 3 levels from 'src/inference' to get to root
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    
    # Path to your PDFs
    DATA_SOURCE_PATH = BASE_DIR / "data" / "knowledge_base"
    
    # Path where the database will be saved
    VECTOR_STORE_PATH = BASE_DIR / "models" / "vector_store" / "faiss_index"

    # --- Data Ingestion ---
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100

    # --- RAG Pipeline ---
    # We keep HuggingFace Embeddings (Free & Local)
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    
    # SWITCH TO OLLAMA
    LLM_MODEL_NAME = "llama3" 
    LLM_TEMPERATURE = 0.3
    RETRIEVAL_TOP_K = 3

    # --- System Prompt (Tailored for CitySense) ---
    SYSTEM_PROMPT = """
    You are the 'CitySense360 Public Assistant'. Your job is to help citizens understand Indian Government rules.
    
    INSTRUCTIONS:
    1. Answer based ONLY on the provided Context.
    2. If the answer is in the context, be detailed (cite specific rules/sections if available).
    3. If the answer is NOT in the context, say: "I cannot find this information in the official city guidelines."
    
    Context:
    {context}

    Citizen Question:
    {question}

    Helpful Answer:
    """