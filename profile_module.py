"""
Module: profile_module.py
Experiment 6 – AI-Powered Medical Symptom Assistant using GPT-4
Part A – User Input (Q1)
Patient profile representation, validation, and sample patient templates.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from utils import SEVERITY_LEVELS


@dataclass
class PatientProfile:
    """
    Data model representing a patient's health profile and reported symptoms.
    Covers all fields required by Part A (Q1).
    """
    age: Optional[int] = None
    sex: str = "Not specified"
    symptoms: List[str] = field(default_factory=list)
    raw_symptoms_text: str = ""
    duration: str = "Not specified"
    severity: str = "Mild"
    medical_history: str = "No known relevant condition"
    current_medications: str = "None reported"
    allergies: str = "No known allergies"

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate patient profile fields.
        Returns (is_valid, list_of_error_messages).
        """
        errors = []

        if self.age is not None:
            if not isinstance(self.age, int) or self.age < 0 or self.age > 125:
                errors.append("Age must be an integer between 0 and 125.")

        if self.severity and self.severity.capitalize() not in [s.capitalize() for s in SEVERITY_LEVELS]:
            errors.append(f"Severity must be one of: {', '.join(SEVERITY_LEVELS)}.")

        return (len(errors) == 0, errors)

    def is_complete(self) -> bool:
        """Check if minimum required fields (Age, Symptoms/Text) are provided."""
        has_symptom_data = bool(self.symptoms) or bool(self.raw_symptoms_text.strip())
        return self.age is not None and has_symptom_data

    def to_dict(self) -> Dict[str, Any]:
        """Convert patient profile to a serializable dictionary."""
        return {
            "age": self.age,
            "sex": self.sex,
            "symptoms": list(self.symptoms),
            "raw_symptoms_text": self.raw_symptoms_text,
            "duration": self.duration,
            "severity": self.severity,
            "medical_history": self.medical_history,
            "current_medications": self.current_medications,
            "allergies": self.allergies,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatientProfile":
        """Instantiate a PatientProfile from a dictionary."""
        age_val = data.get("age")
        if age_val is not None:
            try:
                age_val = int(age_val)
            except (ValueError, TypeError):
                age_val = None

        symptoms_val = data.get("symptoms", [])
        if isinstance(symptoms_val, str):
            symptoms_val = [s.strip() for s in symptoms_val.split(",") if s.strip()]

        return cls(
            age=age_val,
            sex=str(data.get("sex", "Not specified")),
            symptoms=symptoms_val,
            raw_symptoms_text=str(data.get("raw_symptoms_text", "")),
            duration=str(data.get("duration", "Not specified")),
            severity=str(data.get("severity", "Mild")),
            medical_history=str(data.get("medical_history", "No known relevant condition")),
            current_medications=str(data.get("current_medications", "None reported")),
            allergies=str(data.get("allergies", "No known allergies")),
        )

    def format_sample_style(self) -> str:
        """
        Format patient details exactly matching Sample Input in Part A:
        Age : 22
        Sex : Female
        Symptoms : Fever, cough and sore throat
        Duration : 2 days
        Severity : Mild
        History : No known relevant condition
        """
        symptom_str = ", ".join(self.symptoms) if self.symptoms else (self.raw_symptoms_text or "None reported")
        lines = [
            f"Age : {self.age if self.age is not None else 'Not provided'}",
            f"Sex : {self.sex}",
            f"Symptoms : {symptom_str}",
            f"Duration : {self.duration}",
            f"Severity : {self.severity}",
            f"History : {self.medical_history}",
        ]
        if self.current_medications and self.current_medications != "None reported":
            lines.append(f"Current Medications : {self.current_medications}")
        if self.allergies and self.allergies != "No known allergies":
            lines.append(f"Allergies : {self.allergies}")
        return "\n".join(lines)

    def format_summary_card(self) -> str:
        """Produce a formatted ASCII card for terminal display."""
        symptom_str = ", ".join(self.symptoms) if self.symptoms else (self.raw_symptoms_text or "None reported")
        return (
            "┌─────────────────────────────────────────────────────────────┐\n"
            "│ PATIENT PROFILE SUMMARY                                     │\n"
            "├─────────────────────────────────────────────────────────────┤\n"
            f"│ Age                 : {str(self.age if self.age is not None else 'N/A'):<38}│\n"
            f"│ Sex                 : {self.sex:<38}│\n"
            f"│ Reported Symptoms   : {symptom_str[:38]:<38}│\n"
            f"│ Duration            : {self.duration:<38}│\n"
            f"│ Severity            : {self.severity:<38}│\n"
            f"│ Medical History     : {self.medical_history[:38]:<38}│\n"
            f"│ Current Medications : {self.current_medications[:38]:<38}│\n"
            f"│ Known Allergies     : {self.allergies[:38]:<38}│\n"
            "└─────────────────────────────────────────────────────────────┘"
        )


def get_default_sample_profile() -> PatientProfile:
    """
    Return the standard reference sample profile from the lab manual:
    Age: 22, Female, Fever, cough and sore throat, Duration: 2 days, Severity: Mild, History: No known relevant condition.
    """
    return PatientProfile(
        age=22,
        sex="Female",
        symptoms=["Fever", "Cough", "Sore throat"],
        raw_symptoms_text="I have been having fever and cough for two days. My throat is also painful.",
        duration="2 days",
        severity="Mild",
        medical_history="No known relevant condition",
        current_medications="None reported",
        allergies="No known allergies",
    )
