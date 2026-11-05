"""
Module: context_module.py
Experiment 6 – AI-Powered Medical Symptom Assistant using GPT-4
Part C – Context Analysis (Q3)
Analyzes provided patient data and identifies missing clinical information.
"""

import re
from typing import List
from profile_module import PatientProfile
from utils import format_bullet_list, clean_text


class ContextAnalyzer:
    """
    Evaluates reported patient data to determine clinically relevant gaps
    and missing context necessary for a safe differential evaluation.
    """

    @staticmethod
    def identify_missing_information(profile: PatientProfile) -> List[str]:
        """
        Identify missing clinical information based on patient profile and reported symptoms.
        Returns a structured list of missing parameters.
        """
        missing: List[str] = []
        symptoms_str = " ".join(profile.symptoms).lower() + " " + profile.raw_symptoms_text.lower()

        # 1. Exact Temperature Check
        has_fever = any(kw in symptoms_str for kw in ["fever", "temp", "pyrexia", "hot", "burning"])
        has_respiratory_or_infection = any(kw in symptoms_str for kw in ["cough", "throat", "chills", "ache"])
        numeric_temp_match = re.search(
            r"\b(?:(?:9[7-9]|10[0-6])(?:\.\d+)?\s*(?:°?\s*[fF]|degrees?\s*[fF]?)|(?:3[6-9]|4[0-2])(?:\.\d+)?\s*(?:°?\s*[cC]|degrees?\s*[cC]?))\b",
            symptoms_str,
        )
        if (has_fever or has_respiratory_or_infection) and not numeric_temp_match:
            missing.append("Exact temperature")

        # 2. Breathing Difficulty / Respiratory Status Check
        has_respiratory = any(kw in symptoms_str for kw in ["cough", "throat", "cold", "flu", "fever", "chest"])
        has_breathing_mention = any(
            kw in symptoms_str for kw in ["breathing", "breath", "shortness", "dyspnea", "wheezing"]
        )
        if has_respiratory and not has_breathing_mention:
            missing.append("Breathing difficulty")

        # 3. Severity Progression Check
        has_progression = any(
            kw in (profile.raw_symptoms_text + " " + profile.duration).lower()
            for kw in ["worsening", "improving", "better", "worse", "progression", "stable", "unchanged"]
        )
        if not has_progression:
            missing.append("Severity progression")

        # 4. Additional Systemic Symptoms Check (Prioritized as in Q3 sample)
        missing.append("Additional symptoms")

        # 5. Sputum / Cough Characteristics Check
        if "cough" in symptoms_str and not any(kw in symptoms_str for kw in ["dry", "phlegm", "mucus", "productive", "wet"]):
            if "Cough characteristics (dry vs. producing mucus)" not in missing:
                missing.append("Cough characteristics (dry vs. producing mucus)")

        # 6. Specific missing profile fields
        if not profile.duration or profile.duration.lower() in ["not specified", "unknown"]:
            missing.append("Exact onset timing of symptoms")

        if profile.current_medications in ["None reported", "Not specified", ""]:
            missing.append("Any over-the-counter remedies or treatments taken")

        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for item in missing:
            if item.lower() not in seen:
                seen.add(item.lower())
                deduped.append(item)

        return deduped


def analyze_missing_context(profile: PatientProfile) -> List[str]:
    """Convenience helper to identify missing clinical context."""
    return ContextAnalyzer.identify_missing_information(profile)


def format_missing_information(missing_items: List[str]) -> str:
    """
    Format missing information matching Part C (Q3) specification:
    Missing Information:
    • Exact temperature
    • Breathing difficulty
    • Severity progression
    • Additional symptoms
    """
    bullets = format_bullet_list(missing_items)
    return f"Missing Information:\n{bullets}"
