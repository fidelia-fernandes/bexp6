"""
Module: analysis_module.py
Experiment 6 – AI-Powered Medical Symptom Assistant using GPT-4
Part E – Symptom Analysis (Q5) & Part H – Structured Response (Q8)
Coordinates GPT-4 symptom analysis and generates the complete structured report.
Includes both live OpenAI API integration and an intelligent offline clinical engine.
"""

import os
import re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from profile_module import PatientProfile
from symptom_module import SymptomExtractor
from context_module import ContextAnalyzer, format_missing_information
from prompt_module import SYSTEM_PROMPT, construct_analysis_prompt
from question_module import QuestionGenerator, format_follow_up_questions
from safety_module import SafetyEscalationEvaluator, get_disclaimer
from utils import format_bullet_list, format_numbered_list, DISCLAIMER_TEXT

load_dotenv()


class MockMedicalAnalyzer:
    """
    Offline clinical knowledge engine.
    Produces safe, realistic, evidence-informed preliminary symptom analyses
    and differential condition sets when an OpenAI API key is unavailable.
    """

    @staticmethod
    def identify_conditions(symptoms: List[str]) -> List[str]:
        """Map reported symptoms to plausible, non-definitive clinical conditions."""
        symptoms_lower = [s.lower() for s in symptoms]
        joined = " ".join(symptoms_lower)

        # Upper respiratory combination (Matches Part E sample)
        if any(kw in joined for kw in ["fever", "cough", "sore throat", "throat"]):
            return [
                "Common cold",
                "Influenza",
                "Other respiratory infections",
            ]

        # Gastrointestinal symptoms
        if any(kw in joined for kw in ["nausea", "vomit", "diarrhea", "stomach", "abdominal"]):
            return [
                "Viral gastroenteritis",
                "Food-borne illness (Food poisoning)",
                "Acute gastritis",
            ]

        # Allergic symptoms
        if any(kw in joined for kw in ["sneezing", "runny nose", "itchy", "congestion"]):
            return [
                "Allergic rhinitis",
                "Non-allergic rhinitis",
                "Upper respiratory viral prodrome",
            ]

        # Headache / Migraine
        if any(kw in joined for kw in ["headache", "migraine", "head pain"]):
            return [
                "Tension-type headache",
                "Migraine without aura",
                "Sinus-related headache",
            ]

        # Chest / Cardiac red flag
        if any(kw in joined for kw in ["chest pain", "shortness of breath"]):
            return [
                "Musculoskeletal chest wall strain / Costochondritis",
                "Gastroesophageal reflux (GERD)",
                "Acute cardiopulmonary conditions (Emergency evaluation indicated)",
            ]

        # Default general conditions
        return [
            "Non-specific viral syndrome",
            "General fatigue / physical strain",
            "Early stage constitutional illness",
        ]

    @staticmethod
    def generate_explanation(symptoms: List[str], conditions: List[str]) -> str:
        """Provide general, non-diagnostic clinical reasoning."""
        sym_str = ", ".join(symptoms).lower() if symptoms else "reported symptoms"
        cond_str = ", ".join(conditions[:2]) if conditions else "these conditions"

        return (
            f"The reported symptoms ({sym_str}) are frequently observed in several common health conditions, "
            f"most notably {cond_str}. In viral infections, mucosal irritation, systemic immune response, "
            f"and localized inflammation cause overlapping signs such as elevated temperature and respiratory discomfort. "
            f"Because multiple conditions present with nearly identical early characteristics, laboratory investigations, "
            f"physical examination by a clinician, and observation of symptom evolution are necessary to establish "
            f"an accurate diagnosis."
        )


class MedicalSymptomAnalyzer:
    """
    Main controller for symptom analysis and report generation using GPT-4.
    Automatically detects API key presence and falls back gracefully to
    the offline clinical engine when necessary.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.symptom_extractor = SymptomExtractor(api_key=self.api_key, model=self.model)

    def is_live_mode(self) -> bool:
        """Check if an active OpenAI API key is configured."""
        if not self.api_key:
            return False
        if self.api_key.startswith("mock_") or self.api_key.startswith("sk-dummy") or len(self.api_key) < 15:
            return False
        return True

    def get_mode_description(self) -> str:
        """Return human-readable engine status."""
        if self.is_live_mode():
            return f"GPT-4 Live API ({self.model})"
        return "Heuristic Clinical Knowledge Engine (Offline/Mock Mode)"

    def generate_preliminary_analysis(self, profile: PatientProfile) -> str:
        """
        Generate preliminary symptom analysis matching Part E (Q5) format:
        SYMPTOM ANALYSIS
        --------------------------------
        Reported Symptoms:
        • Fever
        • Cough
        • Sore throat
        Possible Conditions:
        • Common cold
        • Influenza
        • Other respiratory infections
        Note:
        These symptoms can occur in multiple conditions.
        The information provided is not sufficient to
        establish a diagnosis.
        """
        symptoms = profile.symptoms
        if not symptoms and profile.raw_symptoms_text:
            symptoms = self.symptom_extractor.extract(profile.raw_symptoms_text)
            profile.symptoms = symptoms

        conditions = MockMedicalAnalyzer.identify_conditions(symptoms)

        # If live API is configured, attempt to get conditions via GPT-4
        if self.is_live_mode():
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.api_key)
                prompt = (
                    f"List 3 possible general medical conditions for a patient with: {', '.join(symptoms)}. "
                    "Respond ONLY with a JSON array of 3 condition names, e.g. [\"Common cold\", \"Influenza\", \"Other respiratory infections\"]."
                )
                res = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a medical informatics AI. Do not diagnose."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=100,
                )
                raw = res.choices[0].message.content.strip()
                import json
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
                parsed = json.loads(raw)
                if isinstance(parsed, list) and len(parsed) >= 2:
                    conditions = parsed[:3]
            except Exception:
                # Keep heuristic conditions
                pass

        bullets_symptoms = format_bullet_list(symptoms)
        bullets_conditions = format_bullet_list(conditions)

        output = (
            "SYMPTOM ANALYSIS\n"
            "--------------------------------\n"
            "Reported Symptoms:\n"
            f"{bullets_symptoms}\n"
            "Possible Conditions:\n"
            f"{bullets_conditions}\n"
            "Note:\n"
            "These symptoms can occur in multiple conditions.\n"
            "The information provided is not sufficient to\n"
            "establish a diagnosis."
        )
        return output

    def generate_structured_response(self, profile: PatientProfile) -> str:
        """
        Generate complete structured response matching Part H (Q8) exact 8-part format:
        1. Reported Symptoms
        2. Patient Information
        3. Possible Conditions
        4. Explanation
        5. Missing Information
        6. Follow-up Questions
        7. Medical Attention Guidance
        8. Disclaimer
        """
        # Ensure symptoms are extracted
        symptoms = profile.symptoms
        if not symptoms and profile.raw_symptoms_text:
            symptoms = self.symptom_extractor.extract(profile.raw_symptoms_text)
            profile.symptoms = symptoms

        # 1. Reported Symptoms
        reported_symptoms_block = format_bullet_list(symptoms)

        # 2. Patient Information
        patient_info_block = (
            f"Age : {profile.age if profile.age is not None else 'Not provided'}\n"
            f"Sex : {profile.sex}\n"
            f"Duration : {profile.duration}\n"
            f"Severity : {profile.severity}\n"
            f"History : {profile.medical_history}\n"
            f"Current Medications : {profile.current_medications}\n"
            f"Allergies : {profile.allergies}"
        )

        # 3. Possible Conditions
        conditions = MockMedicalAnalyzer.identify_conditions(symptoms)
        possible_conditions_block = (
            f"{format_bullet_list(conditions)}\n"
            "Note:\n"
            "These symptoms can occur in multiple conditions.\n"
            "The information provided is not sufficient to\n"
            "establish a diagnosis."
        )

        # 4. Explanation
        explanation_block = MockMedicalAnalyzer.generate_explanation(symptoms, conditions)

        # 5. Missing Information (Part C – Q3)
        missing_items = ContextAnalyzer.identify_missing_information(profile)
        missing_info_block = format_bullet_list(missing_items)

        # 6. Follow-up Questions (Part F – Q6)
        questions = QuestionGenerator.generate(profile, missing_items)
        follow_up_block = format_numbered_list(questions)

        # 7. Medical Attention Guidance (Part G – Q7)
        safety_eval = SafetyEscalationEvaluator.evaluate(profile)
        # Extract the text under the header or clean up
        guidance_text = safety_eval["guidance_text"]
        # Remove the internal "MEDICAL ATTENTION\n--------------------------------\n" if present
        if "MEDICAL ATTENTION\n--------------------------------\n" in guidance_text:
            guidance_content = guidance_text.replace("MEDICAL ATTENTION\n--------------------------------\n", "").strip()
        else:
            guidance_content = guidance_text.strip()

        # 8. Disclaimer (Part I – Q9)
        disclaimer_content = DISCLAIMER_TEXT.replace("DISCLAIMER\n--------------------------------\n", "").strip()

        # If live API is configured, we can enrich the explanation using GPT-4
        if self.is_live_mode():
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.api_key)
                user_prompt = construct_analysis_prompt(profile, symptoms, missing_items)
                res = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                    max_tokens=500,
                )
                ai_text = res.choices[0].message.content.strip()
                # If GPT-4 produced a full response, we can use it or extract explanation
                # To guarantee 100% adherence to Part H 8 numbered sections, we insert its explanation:
                if len(ai_text) > 50:
                    explanation_block = ai_text
            except Exception:
                pass

        # Construct exact 8-part structured response
        response = (
            "======================================================================\n"
            "               STRUCTURED MEDICAL SYMPTOM ANALYSIS                    \n"
            "======================================================================\n\n"
            "1. Reported Symptoms\n"
            "--------------------\n"
            f"{reported_symptoms_block}\n\n"
            "2. Patient Information\n"
            "---------------------\n"
            f"{patient_info_block}\n\n"
            "3. Possible Conditions\n"
            "----------------------\n"
            f"{possible_conditions_block}\n\n"
            "4. Explanation\n"
            "--------------\n"
            f"{explanation_block}\n\n"
            "5. Missing Information\n"
            "----------------------\n"
            f"{missing_info_block}\n\n"
            "6. Follow-up Questions\n"
            "---------------------\n"
            f"{follow_up_block}\n\n"
            "7. Medical Attention Guidance\n"
            "----------------------------\n"
            f"{guidance_content}\n\n"
            "8. Disclaimer\n"
            "-------------\n"
            f"{disclaimer_content}\n\n"
            "======================================================================"
        )
        return response
