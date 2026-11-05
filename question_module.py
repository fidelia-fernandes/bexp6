"""
Module: question_module.py
Experiment 6 – AI-Powered Medical Symptom Assistant using GPT-4
Part F – Follow-Up Questions (Q6)
Generates tailored follow-up questions based on patient profile and reported symptoms.
"""

from typing import List, Optional
from profile_module import PatientProfile
from utils import format_numbered_list


class QuestionGenerator:
    """
    Synthesizes clinically relevant follow-up questions to clarify symptom
    severity, onset, characteristics, and red flags.
    """

    @staticmethod
    def generate(profile: PatientProfile, missing_info: Optional[List[str]] = None) -> List[str]:
        """
        Generate a list of 4-6 targeted follow-up questions.
        """
        questions: List[str] = []
        symptoms_str = " ".join(profile.symptoms).lower() + " " + profile.raw_symptoms_text.lower()

        # 1. Temperature question if fever / infection suspected
        if any(kw in symptoms_str for kw in ["fever", "hot", "pyrexia", "temperature", "chills", "cold", "flu"]):
            questions.append("What is your measured temperature?")

        # 2. Onset timing question
        questions.append("When exactly did the symptoms begin?")

        # 3. Breathing / respiratory question
        if any(kw in symptoms_str for kw in ["cough", "throat", "fever", "chest", "breath"]):
            questions.append("Are you experiencing difficulty breathing?")

        # 4. Cough characteristics
        if "cough" in symptoms_str:
            questions.append("Is the cough dry or producing mucus?")

        # 5. Symptom progression
        questions.append("Are the symptoms improving or worsening?")

        # 6. Throat / swallowing specifics if sore throat present
        if "sore throat" in symptoms_str or "throat" in symptoms_str:
            if len(questions) < 5 or "swallow" not in str(questions):
                # Only add if needed or tailored
                pass

        # 7. Medication question if none reported
        if profile.current_medications in ["None reported", "Not specified", ""] and len(questions) < 6:
            questions.append("Have you taken any over-the-counter medications or home remedies?")

        # 8. Gastrointestinal specific questions
        if any(kw in symptoms_str for kw in ["nausea", "vomit", "diarrhea", "stomach", "abdominal"]):
            if "Are you able to keep liquids down and stay hydrated?" not in questions:
                questions.append("Are you able to keep liquids down and stay hydrated?")

        # Deduplicate while preserving order and limit to top 5-6
        seen = set()
        final_questions = []
        for q in questions:
            if q not in seen:
                seen.add(q)
                final_questions.append(q)

        return final_questions[:6]


def generate_follow_up_questions(
    profile: PatientProfile, missing_info: Optional[List[str]] = None
) -> List[str]:
    """Helper function to generate follow-up questions."""
    return QuestionGenerator.generate(profile, missing_info)


def format_follow_up_questions(questions: List[str]) -> str:
    """
    Format follow-up questions matching Part F (Q6) specification:
    FOLLOW-UP QUESTIONS
    --------------------------------
    1. What is your measured temperature?
    2. When exactly did the symptoms begin?
    3. Are you experiencing difficulty breathing?
    4. Is the cough dry or producing mucus?
    5. Are the symptoms improving or worsening?
    """
    numbered = format_numbered_list(questions)
    return f"FOLLOW-UP QUESTIONS\n--------------------------------\n{numbered}"
