"""
Streamlit Web Dashboard: app_streamlit.py
Experiment 6 – AI-Powered Medical Symptom Assistant using GPT-4
Interactive SaaS Dashboard for clinical symptom analysis and patient guidance.
"""

import os
import sys
import streamlit as st

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from profile_module import PatientProfile, get_default_sample_profile
from input_module import load_sample_cases, get_profile_from_sample_case
from symptom_module import SymptomExtractor, format_extracted_symptoms
from context_module import ContextAnalyzer, format_missing_information
from prompt_module import construct_analysis_prompt, SYSTEM_PROMPT
from question_module import QuestionGenerator, format_follow_up_questions
from safety_module import SafetyEscalationEvaluator, get_disclaimer, get_medical_attention_guidance
from analysis_module import MedicalSymptomAnalyzer
from save_module import save_medical_report, DEFAULT_OUTPUT_FILE, ReportSaver
from utils import DISCLAIMER_TEXT, SEVERITY_LEVELS, RED_FLAG_SYMPTOMS


# ==============================================================================
# PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="Medical Symptom Assistant | GPT-4",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .main-header {
        background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%);
        color: white;
        padding: 24px;
        border-radius: 12px;
        margin-bottom: 24px;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .alert-urgent {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        padding: 14px;
        border-radius: 6px;
        color: #991b1b;
        margin-bottom: 16px;
    }
    .alert-disclaimer {
        background-color: #fefce8;
        border-left: 5px solid #eab308;
        padding: 14px;
        border-radius: 6px;
        color: #854d0e;
        margin-bottom: 16px;
        font-size: 0.9rem;
    }
    .chip {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
        background-color: #e0f2fe;
        color: #0369a1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# STATE INITIALIZATION
# ==============================================================================
if "profile" not in st.session_state:
    st.session_state.profile = get_default_sample_profile()

if "structured_report" not in st.session_state:
    st.session_state.structured_report = ""

if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("OPENAI_API_KEY", "")


# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=64)
    st.title("Medical Assistant")
    st.caption("24CS2506 – Experiment 6 (GPT-4)")

    st.markdown("---")
    st.subheader("⚙️ Configuration")
    api_key_input = st.text_input(
        "OpenAI API Key (Optional)",
        value=st.session_state.api_key,
        type="password",
        help="If left empty, system operates in high-fidelity offline heuristic mode.",
    )
    if api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input

    model_choice = st.selectbox(
        "Model Engine",
        ["gpt-4o", "gpt-4-turbo", "gpt-4"],
        index=0,
    )

    analyzer = MedicalSymptomAnalyzer(api_key=st.session_state.api_key, model=model_choice)
    st.info(f"**Engine:** {analyzer.get_mode_description()}")

    st.markdown("---")
    st.subheader("📋 Preset Sample Cases")
    cases = load_sample_cases()
    case_titles = ["-- Select Preset Case --"] + [f"{c['id']}. {c['title']}" for c in cases]
    selected_case = st.selectbox("Load Sample Case", case_titles)

    if selected_case != "-- Select Preset Case --":
        case_id = int(selected_case.split(".")[0])
        loaded_profile = get_profile_from_sample_case(case_id)
        if loaded_profile and st.button("Apply Selected Case", use_container_width=True):
            st.session_state.profile = loaded_profile
            st.session_state.structured_report = ""
            st.success(f"Loaded Case #{case_id}!")
            st.rerun()

    st.markdown("---")
    st.markdown(
        """
        <div class="alert-disclaimer">
        <strong>⚠️ Medical Notice:</strong><br>
        This tool is strictly informational and educational. It does not replace professional medical diagnosis.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# MAIN PAGE HEADER
# ==============================================================================
st.markdown(
    """
    <div class="main-header">
        <h2>🩺 AI-Powered Medical Symptom Assistant</h2>
        <p>Interactive Clinical Intake, Context Analysis, Follow-up Question Generation, and Escalation Guidance powered by GPT-4</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Patient Status Ribbon
cols = st.columns(4)
with cols[0]:
    st.metric("Age / Sex", f"{st.session_state.profile.age or 'N/A'} yrs / {st.session_state.profile.sex}")
with cols[1]:
    st.metric("Severity", st.session_state.profile.severity)
with cols[2]:
    st.metric("Duration", st.session_state.profile.duration)
with cols[3]:
    safety_check = SafetyEscalationEvaluator.evaluate(st.session_state.profile)
    st.metric("Risk Assessment", safety_check["risk_tier"].split("/")[0].strip())


# ==============================================================================
# TABS INTERFACE
# ==============================================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "1. Patient Intake & Symptoms",
        "2. Context Analysis",
        "3. GPT-4 Prompt",
        "4. Symptom Analysis",
        "5. Follow-Up Questions",
        "6. Safety & Guidance",
        "7. Save & Export Report",
    ]
)

# ------------------------------------------------------------------------------
# TAB 1: PATIENT INTAKE (Part A & B)
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Part A & B – Patient Health Profile & Symptoms")

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("#### Patient Demographics & History (Q1)")
        age_val = st.number_input("Age", min_value=0, max_value=125, value=st.session_state.profile.age or 22)
        sex_val = st.selectbox(
            "Sex", ["Female", "Male", "Other", "Not specified"],
            index=["Female", "Male", "Other", "Not specified"].index(st.session_state.profile.sex)
            if st.session_state.profile.sex in ["Female", "Male", "Other", "Not specified"] else 0
        )
        duration_val = st.text_input("Duration of Symptoms", value=st.session_state.profile.duration)
        severity_val = st.selectbox(
            "Severity", SEVERITY_LEVELS,
            index=SEVERITY_LEVELS.index(st.session_state.profile.severity)
            if st.session_state.profile.severity in SEVERITY_LEVELS else 0
        )
        history_val = st.text_input("Relevant Medical History", value=st.session_state.profile.medical_history)
        meds_val = st.text_input("Current Medications", value=st.session_state.profile.current_medications)
        allergies_val = st.text_input("Known Allergies", value=st.session_state.profile.allergies)

    with col_b:
        st.markdown("#### Natural-Language Symptoms Narrative (Q2)")
        narrative_val = st.text_area(
            "Describe symptoms in patient's natural words",
            value=st.session_state.profile.raw_symptoms_text,
            height=130,
        )

        if st.button("Extract Symptoms & Update Profile", type="primary", use_container_width=True):
            st.session_state.profile.age = age_val
            st.session_state.profile.sex = sex_val
            st.session_state.profile.duration = duration_val
            st.session_state.profile.severity = severity_val
            st.session_state.profile.medical_history = history_val
            st.session_state.profile.current_medications = meds_val
            st.session_state.profile.allergies = allergies_val
            st.session_state.profile.raw_symptoms_text = narrative_val

            extracted = analyzer.symptom_extractor.extract(narrative_val)
            st.session_state.profile.symptoms = extracted
            st.session_state.structured_report = ""
            st.success("Symptoms successfully extracted and profile updated!")

        st.markdown("#### Extracted Clinical Symptoms (Part B – Q2):")
        if st.session_state.profile.symptoms:
            for s in st.session_state.profile.symptoms:
                st.markdown(f"<span class='chip'>• {s}</span>", unsafe_allow_html=True)
        else:
            st.info("No symptoms extracted yet. Click 'Extract Symptoms & Update Profile' above.")

# ------------------------------------------------------------------------------
# TAB 2: CONTEXT ANALYSIS (Part C – Q3)
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Part C – Context Analysis & Missing Information (Q3)")
    missing = ContextAnalyzer.identify_missing_information(st.session_state.profile)

    st.markdown("The system analyzes the available clinical context and flags relevant details not yet reported:")
    st.code(format_missing_information(missing), language="markdown")

    st.info(
        "💡 Clinical Context Detection ensures that critical parameters like objective fever readings, "
        "respiratory compromise, and progression timelines are highlighted before a differential is formulated."
    )

# ------------------------------------------------------------------------------
# TAB 3: PROMPT GENERATION (Part D – Q4)
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Part D – Prompt Construction for GPT-4 (Q4)")
    st.markdown(
        "The constructed prompt enforces strict non-diagnostic boundaries, reasoning requirements, "
        "and follow-up directives for GPT-4:"
    )

    extracted = st.session_state.profile.symptoms
    missing = ContextAnalyzer.identify_missing_information(st.session_state.profile)
    user_prompt = construct_analysis_prompt(st.session_state.profile, extracted, missing)

    with st.expander("View System Prompt (Guardrails & Instructions)", expanded=False):
        st.code(SYSTEM_PROMPT, language="markdown")

    with st.expander("View Generated User Prompt", expanded=True):
        st.code(user_prompt, language="markdown")

# ------------------------------------------------------------------------------
# TAB 4: SYMPTOM ANALYSIS (Part E – Q5 & Part H – Q8)
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("Part E – Symptom Analysis & Part H – Structured Response")

    if st.button("🚀 Run Complete Analysis with GPT-4", type="primary"):
        with st.spinner("Analyzing symptoms and synthesizing clinical differential..."):
            st.session_state.structured_report = analyzer.generate_structured_response(st.session_state.profile)

    if st.session_state.structured_report:
        st.markdown("#### Complete 8-Part Structured Response (Part H – Q8)")
        st.code(st.session_state.structured_report, language="markdown")
    else:
        # Show Preliminary Sample Output
        st.markdown("#### Preliminary Symptom Analysis Preview (Part E – Q5):")
        preliminary = analyzer.generate_preliminary_analysis(st.session_state.profile)
        st.code(preliminary, language="markdown")

# ------------------------------------------------------------------------------
# TAB 5: FOLLOW-UP QUESTIONS (Part F – Q6)
# ------------------------------------------------------------------------------
with tab5:
    st.subheader("Part F – Follow-Up Questions (Q6)")
    missing = ContextAnalyzer.identify_missing_information(st.session_state.profile)
    questions = QuestionGenerator.generate(st.session_state.profile, missing)

    st.markdown("Generated clarifying clinical questions:")
    st.code(format_follow_up_questions(questions), language="markdown")

# ------------------------------------------------------------------------------
# TAB 6: SAFETY & ESCALATION (Part G – Q7 & Part I – Q9)
# ------------------------------------------------------------------------------
with tab6:
    st.subheader("Part G & I – Medical Attention Guidance & Disclaimer")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("#### Escalation Guidance (Part G – Q7)")
        guidance = get_medical_attention_guidance(st.session_state.profile)
        st.code(guidance, language="markdown")

    with col_g2:
        st.markdown("#### Standard Disclaimer (Part I – Q9)")
        st.code(get_disclaimer(), language="markdown")

# ------------------------------------------------------------------------------
# TAB 7: SAVE & EXPORT (Part J – Q10)
# ------------------------------------------------------------------------------
with tab7:
    st.subheader("Part J – Save Analysis to File (Q10)")
    st.markdown(f"Destination: `{DEFAULT_OUTPUT_FILE}`")

    if not st.session_state.structured_report:
        st.session_state.structured_report = analyzer.generate_structured_response(st.session_state.profile)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("💾 Save Report to outputs/medical_analysis.txt", use_container_width=True):
            ok, msg = save_medical_report(st.session_state.structured_report)
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    with col_s2:
        st.download_button(
            label="📥 Download medical_analysis.txt",
            data=st.session_state.structured_report,
            file_name="medical_analysis.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.markdown("#### Report Content Preview:")
    st.text_area("Report Preview", value=st.session_state.structured_report, height=350)
