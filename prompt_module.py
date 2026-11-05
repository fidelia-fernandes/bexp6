"""
Module: prompt_module.py
Experiment 6 – AI-Powered Medical Symptom Assistant using GPT-4
Part D – Prompt Generation (Q4)
Constructs medically sound, safety-constrained prompts for GPT-4.
"""

from typing import List, Dict, Any, Optional
from profile_module import PatientProfile


# System prompt with strict instructions adhering to Q4 constraints
SYSTEM_PROMPT = """You are an AI-powered Medical Symptom Assistant designed to provide informational and educational support.
You are NOT a doctor, and your responses MUST NOT constitute a medical diagnosis or treatment plan.

You must strictly adhere to the following principles:
1. ANALYZE SYMPTOMS: Carefully examine the reported symptoms in conjunction with the patient's age, duration, and medical history.
2. IDENTIFY POSSIBLE CONDITIONS: List 3 to 5 common, plausible conditions associated with the reported symptoms.
3. EXPLAIN REASONING AT A GENERAL LEVEL: Provide clear, non-technical explanations of why these symptoms can be associated with the conditions, noting overlapping features.
4. ASK FOLLOW-UP QUESTIONS: Generate 4 to 6 relevant clinical follow-up questions that would help clarify the condition (e.g., onset, progression, severity, measured vitals).
5. IDENTIFY WHEN PROFESSIONAL MEDICAL CARE IS APPROPRIATE: Clearly outline red flags, warning signs, and circumstances requiring prompt or urgent clinical evaluation.
6. AVOID DEFINITIVE DIAGNOSIS: Never state or imply that the patient definitely has a specific disease. Use cautious phrasing such as "Possible conditions include...", "These symptoms are commonly seen in...", and "Only a healthcare provider can establish a diagnosis."
7. AVOID INVENTING INFORMATION: Do not hallucinate symptoms, medications, or test results not provided by the user. If information is missing, explicitly note it as missing.

Your output must be structured, cautious, empathetic, and objective.
"""


def construct_analysis_prompt(
    profile: PatientProfile,
    extracted_symptoms: Optional[List[str]] = None,
    missing_info: Optional[List[str]] = None,
) -> str:
    """
    Construct user prompt for GPT-4 incorporating patient profile,
    extracted symptoms, and context analysis. Adheres to Part D (Q4).
    """
    symptoms_list = extracted_symptoms if extracted_symptoms is not None else profile.symptoms
    symptoms_display = ", ".join(symptoms_list) if symptoms_list else (profile.raw_symptoms_text or "None reported")

    missing_section = ""
    if missing_info:
        missing_section = "\nPotentially Missing Clinical Information Identified:\n" + "\n".join(
            [f"- {m}" for m in missing_info]
        )

    user_prompt = f"""Please analyze the following patient health information and symptoms:

PATIENT INFORMATION:
- Age: {profile.age if profile.age is not None else 'Not provided'}
- Sex: {profile.sex}
- Duration: {profile.duration}
- Stated Severity: {profile.severity}
- Medical History: {profile.medical_history}
- Current Medications: {profile.current_medications}
- Known Allergies: {profile.allergies}

REPORTED / EXTRACTED SYMPTOMS:
- {symptoms_display}

NATURAL LANGUAGE PATIENT DESCRIPTION:
"{profile.raw_symptoms_text or symptoms_display}"
{missing_section}

INSTRUCTIONS FOR YOUR RESPONSE:
Please provide your analysis strictly following this format:
1. Reported Symptoms (bullet list)
2. Possible Conditions (3-5 common possibilities with note on differential overlap)
3. Explanation (general reasoning connecting symptoms to conditions)
4. Missing Information (key details not yet known)
5. Follow-up Questions (numbered list of 4-5 relevant clarifying questions)
6. Medical Attention Guidance (cautious advice on when to seek professional care)
7. Disclaimer (statement that this is informational and not a diagnosis)

Remember: Avoid any definitive diagnosis, and do not invent any unstated details.
"""
    return user_prompt.strip()


def construct_symptom_extraction_prompt(user_text: str) -> str:
    """Construct prompt for extracting symptoms from natural language text."""
    return f"""Extract all medical symptoms mentioned in the following user statement.
Standardize them into concise clinical terms (e.g., "throat is painful" -> "Sore throat", "fever" -> "Fever").

User Statement:
"{user_text}"

Respond ONLY with a valid JSON array of strings, for example: ["Fever", "Cough", "Sore throat"].
Do not add any additional explanation or formatting.
"""


def get_prompt_overview(profile: PatientProfile, extracted: List[str], missing: List[str]) -> Dict[str, str]:
    """Return dictionary with system and user prompts for inspection."""
    return {
        "system_prompt": SYSTEM_PROMPT.strip(),
        "user_prompt": construct_analysis_prompt(profile, extracted, missing),
    }
