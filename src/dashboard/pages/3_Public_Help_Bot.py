import streamlit as st
import os
import sys
import time
from langchain_core.prompts import PromptTemplate

# --- CRITICAL FIX: IMPORT FROM CLASSIC ---
try:
    # New location for LangChain v1.0+
    from langchain_classic.chains import RetrievalQA
except ImportError:
    try:
        # Standard location for older versions
        from langchain.chains import RetrievalQA
    except ImportError:
        # Last resort fallback
        from langchain_community.chains import RetrievalQA

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama

from langchain_community.vectorstores import FAISS

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.inference.config import Config

# --- PAGE CONFIG ---
st.set_page_config(page_title="Public Help Bot", page_icon="🤖", layout="wide")

st.title("🤖 Public Help Assistant")
st.markdown("### *Ask about Traffic Rules, Swachh Bharat & Safety Guidelines for Emergency*")
st.divider()

# --- LOAD ENGINE ---
@st.cache_resource
def load_rag_pipeline():
    # 1. Load Vector Store
    if not os.path.exists(Config.VECTOR_STORE_PATH):
        return None
        
    # Load Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    
    try:
        # Load FAISS DB
        db = FAISS.load_local(
            str(Config.VECTOR_STORE_PATH), 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        retriever = db.as_retriever(search_kwargs={'k': Config.RETRIEVAL_TOP_K})
        
        # 2. Load Ollama (Llama 3)
        llm = ChatOllama(
            model=Config.LLM_MODEL_NAME,
            temperature=Config.LLM_TEMPERATURE
        )

        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=Config.SYSTEM_PROMPT
        )

        # Create Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )
        return qa_chain
        
    except Exception as e:
        st.error(f"Error loading DB: {e}")
        return None

qa_chain = load_rag_pipeline()

# --- CHAT UI ---
if qa_chain is None:
    st.warning("⚠️ Knowledge Base not found.")
    st.info("Run `python src/inference/ingest.py` to load your PDFs first!")
else:
    # Initialize Chat History
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = [
            {"role": "assistant", "content": "Hello! I have read the Motor Vehicles Act and other city guidelines. Ask me anything!"}
        ]

    # Display History
    for msg in st.session_state.rag_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User Input
    if prompt := st.chat_input("Ex: What is the fine for overspeeding?"):
        # Show User Message
        st.session_state.rag_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Answer
        with st.chat_message("assistant"):
            with st.spinner("📖 Reading official documents..."):
                start = time.time()
                try:
                    response = qa_chain.invoke({"query": prompt})
                    latency = time.time() - start
                    
                    result_text = response['result']
                    
                    # Append Sources
                    result_text += "\n\n---\n**📚 Sources Used:**\n"
                    seen_sources = set()
                    if 'source_documents' in response:
                        for doc in response['source_documents']:
                            fname = os.path.basename(doc.metadata.get('source', 'Doc'))
                            page = doc.metadata.get('page', '?')
                            src = f"{fname} (Page {page})"
                            if src not in seen_sources:
                                result_text += f"- *{src}*\n"
                                seen_sources.add(src)
                    
                    result_text += f"\n*Generated in {latency:.2f}s*"
                    st.markdown(result_text)
                    
                    # Save Assistant Message
                    st.session_state.rag_messages.append({"role": "assistant", "content": result_text})
                    
                except Exception as e:
                    st.error(f"RAG Error: {e}")