import torch
import joblib
import os
import sys
import json
import time
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# --- CONFIG & PATHS ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
MODEL_PATH = os.path.join(BASE_DIR, "models/complaints_bert/")
ENCODER_PATH = os.path.join(BASE_DIR, "models/complaints_label_encoder.pkl")
REPORT_DIR = os.path.join(BASE_DIR, "reports/")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. LOAD BERT (THE CLASSIFIER TOOL) ---
print("🤖 Loading Neural Classifier...")
try:
    _tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
    _bert_model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
    _label_encoder = joblib.load(ENCODER_PATH)
    _bert_model.eval()
    print("   ✅ DistilBERT Loaded.")
except Exception as e:
    print(f"   ⚠️ BERT Load Error: {e}")
    _bert_model = None

# --- 2. DEFINE THE TOOL FUNCTION ---
def classify_complaint_tool(text: str) -> str:
    """Uses the BERT model to predict the department."""
    if not _bert_model: 
        return "General Administration"
    
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        outputs = _bert_model(**inputs)
    
    pred_idx = torch.argmax(outputs.logits, dim=1).item()
    dept = _label_encoder.inverse_transform([pred_idx])[0]
    return dept

# --- 3. LLM SETUP (OLLAMA) ---
llm = ChatOllama(
    model="llama3", 
    format="json",   
    temperature=0.1 
)

# --- HELPER: FORCE TEXT (Prevents TypeErrors) ---
def force_text(val):
    if val is None: return "N/A"
    if isinstance(val, dict): return ". ".join([str(v) for v in val.values()])
    if isinstance(val, list): return ". ".join([str(v) for v in val])
    return str(val)

# --- 4. THE AGENT LOGIC ---
def run_complaint_agent(complaint_text, user_name="Citizen"):
    """
    Orchestrates: Classify -> Reason -> Report (Original + Response + Work Order)
    """
    
    # STEP 1: EXECUTE TOOL (Classify)
    department = classify_complaint_tool(complaint_text)
    
    # STEP 2: CONSTRUCT PROMPT
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a Smart City Operations AI. Return ONLY JSON.
        
        TASK:
        1. Analyze the complaint.
        2. Generate TWO outputs:
           - A polite notification for the Citizen.
           - A technical INTERNAL WORK ORDER for the Engineering Team.
        
        REQUIRED JSON STRUCTURE:
        {{
            "summary": "1-sentence technical summary.",
            "severity": "High/Medium/Low",
            "department": "The Classified Department",
            "citizen_response": "Dear Citizen... [Polite acknowledgment]...",
            "internal_action_plan": "WORK ORDER ID: [Auto]\\nISSUE: [Tech Details]\\nLOCATION: [Extract]\\nACTION: [Steps]"
        }}
        """),
        ("human", """
        COMPLAINT: "{text}"
        CITIZEN NAME: "{name}"
        CLASSIFIED DEPT: "{dept}"
        """)
    ])
    
    # STEP 3: AI GENERATION
    try:
        response = llm.invoke([
            HumanMessage(content=final_prompt.format(
                text=complaint_text,
                name=user_name,
                dept=department
            ))
        ])
        ai_data = json.loads(response.content)
    except Exception as e:
        return {"error": f"AI Generation Failed: {str(e)}"}

    # STEP 4: SAVE REPORT TO FILE (FULL RECORD)
    os.makedirs(REPORT_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    safe_dept = force_text(ai_data.get('department', 'General')).replace(" ", "_").replace("/", "_")
    filename = f"Report_{safe_dept}_{timestamp}.txt"
    filepath = os.path.join(REPORT_DIR, filename)
    
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("==================================================\n")
            f.write("          CITYSENSE360 - AUTOMATED DOCKET         \n")
            f.write("==================================================\n")
            f.write(f"DATE: {timestamp}\n")
            f.write(f"REF ID: CS360-{int(time.time())}\n\n")
            
            # SECTION 1: THE ORIGINAL COMPLAINT (Added per your request)
            f.write("SECTION 1: ORIGINAL GRIEVANCE FILED\n")
            f.write("-" * 30 + "\n")
            f.write(f"CITIZEN:   {user_name}\n")
            f.write(f"COMPLAINT: {complaint_text}\n\n")
            
            # SECTION 2: RESPONSE TO CITIZEN
            f.write("SECTION 2: CITIZEN NOTIFICATION\n")
            f.write("-" * 30 + "\n")
            f.write(force_text(ai_data.get('citizen_response')))
            f.write("\n\n")
            
            # SECTION 3: INTERNAL WORK ORDER
            f.write("SECTION 3: INTERNAL OPERATIONS (WORK ORDER)\n")
            f.write("-" * 30 + "\n")
            f.write(f"DEPARTMENT: {force_text(ai_data.get('department'))}\n")
            f.write(f"SEVERITY:   {force_text(ai_data.get('severity'))}\n")
            f.write(f"SUMMARY:    {force_text(ai_data.get('summary'))}\n\n")
            f.write("TECHNICAL ACTION PLAN:\n")
            f.write(force_text(ai_data.get('internal_action_plan')))
            
    except Exception as e:
        return {"error": f"File Write Error: {str(e)}"}

    # Return data to Dashboard
    ai_data["report_path"] = filepath
    return ai_data