"""
Module: safety_module.py
Experiment 6 – AI-Powered Medical Symptom Assistant using GPT-4
Part G – Risk / Escalation Assessment (Q7) & Part I – Disclaimer (Q9)
Safety boundaries, clinical escalation assessment, and medical disclaimer.
"""

from typing import Dict, Any, List, Optional
from profile_module import PatientProfile
from utils import DISCLAIMER_TEXT, RED_FLAG_SYMPTOMS


class SafetyEscalationEvaluator:
    """
    Evaluates reported symptoms and patient profile to determine appropriate
    escalation guidance using cautious, non-triage language.
    """

    @staticmethod
    def evaluate(profile: PatientProfile) -> Dict[str, Any]:
        """
        Evaluate risk level and provide cautious medical attention guidance.
        """
        symptoms_text = (
            " ".join(profile.symptoms) + " " + profile.raw_symptoms_text
        ).lower()

        # Check for red flags
        detected_red_flags: List[str] = []
        for rf in RED_FLAG_SYMPTOMS:
            if rf in symptoms_text:
                detected_red_flags.append(rf.title())

        severity = (profile.severity or "Mild").capitalize()
        is_urgent = bool(detected_red_flags) or severity in ["Severe", "Critical"]
        is_moderate = severity == "Moderate" or "day" in profile.duration.lower() and any(
            c in profile.duration for c in ["5", "6", "7", "week", "month"]
        )

        # Build cautious guidance
        if is_urgent:
            guidance_text = (
                "MEDICAL ATTENTION\n"
                "--------------------------------\n"
                "If symptoms are severe, rapidly worsening,\n"
                "or concerning, seek appropriate professional\n"
                "medical care.\n\n"
                "CRITICAL WARNING:\n"
                "You have reported symptoms that may require urgent medical evaluation\n"
                f"(Flags noted: {', '.join(detected_red_flags) if detected_red_flags else severity}).\n"
                "Please do not wait. Contact local emergency services or visit an emergency room immediately."
            )
            risk_tier = "Urgent / High"
        elif is_moderate:
            guidance_text = (
                "MEDICAL ATTENTION\n"
                "--------------------------------\n"
                "If symptoms are severe, rapidly worsening,\n"
                "or concerning, seek appropriate professional\n"
                "medical care.\n\n"
                "Consider scheduling an appointment with a healthcare professional or visiting an urgent care clinic\n"
                "if your symptoms persist for more than 3-5 days, prevent normal hydration, or progressively worsen."
            )
            risk_tier = "Moderate / Clinical Consultation Advised"
        else:
            # Matches exact Part G example
            guidance_text = (
                "MEDICAL ATTENTION\n"
                "--------------------------------\n"
                "If symptoms are severe, rapidly worsening,\n"
                "or concerning, seek appropriate professional\n"
                "medical care."
            )
            risk_tier = "Low / Informational Routine"

        return {
            "risk_tier": risk_tier,
            "detected_red_flags": detected_red_flags,
            "guidance_text": guidance_text,
            "disclaimer": DISCLAIMER_TEXT,
        }


def get_medical_attention_guidance(profile: PatientProfile) -> str:
    """Helper to get formatted medical attention guidance (Part G – Q7)."""
    eval_result = SafetyEscalationEvaluator.evaluate(profile)
    return eval_result["guidance_text"]


def get_disclaimer() -> str:
    """Helper to return the exact medical disclaimer (Part I – Q9)."""
    return DISCLAIMER_TEXT
