import streamlit as st
import os
import sys

# --- SETUP PATHS ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.agent.complaint_agent import run_complaint_agent, force_text 

# --- PAGE CONFIG ---
st.set_page_config(page_title="Citizen Grievance Portal", page_icon="📢", layout="wide")

st.title("📢 Smart City Grievance Portal")
st.markdown("### *Agentic AI Resolution System*")
st.divider()

# --- SESSION STATE SETUP (Fixes the "Disappearing" Issue) ---
if 'complaint_result' not in st.session_state:
    st.session_state.complaint_result = None

# --- UI LAYOUT ---
col_input, col_output = st.columns([1, 1])

with col_input:
    st.subheader("📝 File a Complaint")
    with st.container(border=True):
        name = st.text_input("Citizen Name", "Anonymous")
        # If we have a result, we can keep the text, or clear it. 
        # Here we keep it so the user sees what they typed.
        complaint = st.text_area("Describe the Issue", height=200, 
                               placeholder="Example: Large pothole on 4th Main Road...")
        
        c1, c2 = st.columns(2)
        with c1:
            submit = st.button("🚀 Process Grievance", type="primary", use_container_width=True)
        with c2:
            # The "New Complaint" button clears the memory
            if st.button("🔄 New Complaint", use_container_width=True):
                st.session_state.complaint_result = None
                st.rerun()

with col_output:
    st.subheader("🤖 Agent Actions")
    
    # CASE 1: USER CLICKS SUBMIT
    if submit and complaint:
        with st.spinner("🔄 Agent is classifying & drafting report..."):
            # Run Agent
            result = run_complaint_agent(complaint, name)
            # SAVE to Session State so it survives download clicks
            st.session_state.complaint_result = result
            st.rerun() # Force a refresh to show the data immediately

    # CASE 2: DATA EXISTS IN MEMORY (This runs after download clicks too)
    if st.session_state.complaint_result:
        result = st.session_state.complaint_result
        
        if "error" in result:
            st.error(f"Agent Failed: {result['error']}")
        else:
            # Display Success Card
            st.success("✅ Case Processed Successfully")
            
            # --- DEPARTMENT MAPPING (Converts Codes to Names) ---
            dept_mapping = {
                "GOVGJ": "Sanitation & Waste Mgmt",
                "MCD": "Municipal Corporation",
                "NHAI": "Roads & Highways",
                "BESCOM": "Energy Department",
                "BSNL": "Telecommunications",
                "PMOPG": "Public Infrastructure (PMO)",
                "DPOST": "Postal Services",
                "MOLBB": "Labor & Employment",
                "MORLY": "Railways",
                "DEABD": "Economic Affairs"
            }

            raw_dept = force_text(result.get("department"))
            # If code is found in dictionary, use the nice name. Else, keep the original code.
            display_dept = dept_mapping.get(raw_dept, raw_dept)
            sev_display = force_text(result.get("severity"))

            with st.container(border=True):
                c1, c2 = st.columns(2)
                
                # Display mapped department name
                c1.metric("Department", display_dept)
                c2.metric("Severity", sev_display)
                
                st.markdown("**Summary:**")
                st.info(force_text(result.get("summary")))
                
                # VISUAL TABS (Matches the Text File Content)
                tab1, tab2 = st.tabs(["📧 Citizen Response", "🛠️ Internal Work Order"])
                
                with tab1:
                    st.text_area("Draft to Citizen", force_text(result.get("citizen_response")), height=250)
                
                with tab2:
                    st.caption("Technical Action Plan (JSON/YAML View):")
                    # Displaying the Work Order text clearly
                    plan_text = force_text(result.get("internal_action_plan"))
                    st.text_area("Work Order Details", plan_text, height=250)
                
                # Download Button
                if os.path.exists(result["report_path"]):
                    with open(result["report_path"], "r") as f:
                        file_txt = f.read()
                        
                    st.download_button(
                        label="📥 Download Official Record (.txt)",
                        data=file_txt,
                        file_name=os.path.basename(result["report_path"]),
                        mime="text/plain",
                        type="primary"
                    )

    # CASE 3: NOTHING HAPPENED YET
    elif not st.session_state.complaint_result:
        st.info("👈 Enter a complaint and click Process to start the AI Agent.")