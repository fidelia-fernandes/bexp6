"""
Module: input_module.py
Experiment 6 – AI-Powered Medical Symptom Assistant using GPT-4
Part A – User Input (Q1) & Part B – Symptom Input (Q2)
Collects patient demographic information, medical history, and natural-language symptoms.
"""

import os
import csv
from typing import Optional, List, Dict, Any

from profile_module import PatientProfile, get_default_sample_profile
from utils import SEVERITY_LEVELS, print_section


SAMPLE_CASES_CSV = os.path.join(os.path.dirname(__file__), "data", "sample_cases.csv")


def load_sample_cases() -> List[Dict[str, Any]]:
    """Load sample patient cases from data/sample_cases.csv."""
    cases = []
    if not os.path.exists(SAMPLE_CASES_CSV):
        return cases

    try:
        with open(SAMPLE_CASES_CSV, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cases.append(row)
    except Exception as e:
        print(f"[!] Warning: Unable to load sample cases: {e}")

    return cases


def get_profile_from_sample_case(case_id: int) -> Optional[PatientProfile]:
    """Retrieve a pre-configured sample profile by ID from sample_cases.csv."""
    cases = load_sample_cases()
    for row in cases:
        if str(row.get("id")) == str(case_id):
            symptoms_list = [s.strip() for s in row.get("symptoms", "").split(",") if s.strip()]
            return PatientProfile(
                age=int(row.get("age", 22)),
                sex=row.get("sex", "Female"),
                symptoms=symptoms_list,
                raw_symptoms_text=row.get("raw_narrative", ""),
                duration=row.get("duration", "2 days"),
                severity=row.get("severity", "Mild"),
                medical_history=row.get("medical_history", "No known relevant condition"),
                current_medications=row.get("current_medications", "None reported"),
                allergies=row.get("allergies", "No known allergies"),
            )
    return None


class PatientInputCollector:
    """CLI Input Handler for patient data collection (Part A – Q1)."""

    @staticmethod
    def collect_patient_profile(existing: Optional[PatientProfile] = None) -> PatientProfile:
        """
        Interactively collect all patient fields specified in Q1:
        - Age
        - Sex, if relevant
        - Duration
        - Severity
        - Relevant medical history
        - Current medications, if relevant
        - Allergies, if relevant
        """
        print("\n" + "=" * 60)
        print("          PATIENT INFORMATION COLLECTION (PART A)          ")
        print("=" * 60)
        print("Please enter patient details (press Enter to keep default):")

        # Age
        curr_age = existing.age if existing and existing.age is not None else 22
        while True:
            age_input = input(f"Age [{curr_age}]: ").strip()
            if not age_input:
                age = curr_age
                break
            try:
                age = int(age_input)
                if 0 <= age <= 125:
                    break
                print("[!] Please enter a valid age between 0 and 125.")
            except ValueError:
                print("[!] Invalid age. Please enter a whole number.")

        # Sex
        curr_sex = existing.sex if existing and existing.sex else "Female"
        print("Sex: [1] Female  [2] Male  [3] Other  [4] Prefer not to say")
        sex_choice = input(f"Select sex (1-4 or type value) [{curr_sex}]: ").strip()
        if not sex_choice:
            sex = curr_sex
        elif sex_choice == "1":
            sex = "Female"
        elif sex_choice == "2":
            sex = "Male"
        elif sex_choice == "3":
            sex = "Other"
        elif sex_choice == "4":
            sex = "Not specified"
        else:
            sex = sex_choice

        # Duration
        curr_dur = existing.duration if existing and existing.duration else "2 days"
        dur_input = input(f"Duration of symptoms (e.g., 2 days, 1 week) [{curr_dur}]: ").strip()
        duration = dur_input if dur_input else curr_dur

        # Severity
        curr_sev = existing.severity if existing and existing.severity else "Mild"
        print(f"Severity options: {', '.join(SEVERITY_LEVELS)}")
        sev_input = input(f"Severity level [{curr_sev}]: ").strip()
        if not sev_input:
            severity = curr_sev
        else:
            matched = [s for s in SEVERITY_LEVELS if s.lower() == sev_input.lower()]
            severity = matched[0] if matched else sev_input.capitalize()

        # Medical History
        curr_hist = existing.medical_history if existing and existing.medical_history else "No known relevant condition"
        hist_input = input(f"Relevant medical history [{curr_hist}]: ").strip()
        medical_history = hist_input if hist_input else curr_hist

        # Current Medications
        curr_meds = existing.current_medications if existing and existing.current_medications else "None reported"
        meds_input = input(f"Current medications, if relevant [{curr_meds}]: ").strip()
        current_medications = meds_input if meds_input else curr_meds

        # Allergies
        curr_alg = existing.allergies if existing and existing.allergies else "No known allergies"
        alg_input = input(f"Known allergies, if relevant [{curr_alg}]: ").strip()
        allergies = alg_input if alg_input else curr_alg

        # Retain existing symptoms if present
        symptoms = existing.symptoms if existing else []
        raw_text = existing.raw_symptoms_text if existing else ""

        profile = PatientProfile(
            age=age,
            sex=sex,
            symptoms=symptoms,
            raw_symptoms_text=raw_text,
            duration=duration,
            severity=severity,
            medical_history=medical_history,
            current_medications=current_medications,
            allergies=allergies,
        )

        print("\n[OK] Patient profile successfully updated!")
        return profile

    @staticmethod
    def collect_symptoms_narrative(existing: Optional[PatientProfile] = None) -> str:
        """
        Collect natural-language symptoms input from the user (Part B – Q2).
        Example: "I have been having fever and cough for two days. My throat is also painful."
        """
        print("\n" + "=" * 60)
        print("          NATURAL-LANGUAGE SYMPTOM INPUT (PART B)          ")
        print("=" * 60)
        default_narrative = (
            "I have been having fever and cough for two days. My throat is also painful."
        )
        print("Describe your symptoms naturally in your own words.")
        print(f"Example: \"{default_narrative}\"")
        print("-" * 60)

        user_input = input("\nEnter symptoms narrative (press Enter to use lab sample): ").strip()
        if not user_input:
            user_input = default_narrative
            print(f"[i] Using default sample symptoms: \"{user_input}\"")

        return user_input

    @staticmethod
    def display_sample_cases_menu() -> Optional[PatientProfile]:
        """Display loaded sample cases from data/sample_cases.csv and allow quick selection."""
        cases = load_sample_cases()
        if not cases:
            print("[!] No sample cases found in data/sample_cases.csv.")
            return None

        print("\n" + "=" * 68)
        print("                    SELECT A PRESET SAMPLE CASE                     ")
        print("=" * 68)
        for c in cases:
            print(f"  [{c['id']}] {c['title']} ({c['age']}yo {c['sex']})")
            print(f"      Symptoms: {c['symptoms']} | Duration: {c['duration']} | Severity: {c['severity']}")
        print("  [0] Cancel / Keep current profile")
        print("-" * 68)

        choice = input("Enter case number (0-5): ").strip()
        if choice in ["", "0"]:
            return None

        profile = get_profile_from_sample_case(int(choice)) if choice.isdigit() else None
        if profile:
            print(f"\n[OK] Loaded preset case: Case #{choice} ({profile.sex}, Age {profile.age})")
            return profile
        else:
            print("[!] Invalid case number.")
            return None
