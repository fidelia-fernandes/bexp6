"""
Unit & Integration Test Suite for Experiment 6
Validates compliance with Parts A through K and Questions Q1 to Q11.
"""

import os
import sys
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profile_module import PatientProfile, get_default_sample_profile
from symptom_module import extract_symptoms_from_text, format_extracted_symptoms, SymptomExtractor
from context_module import analyze_missing_context, format_missing_information, ContextAnalyzer
from prompt_module import construct_analysis_prompt, SYSTEM_PROMPT
from question_module import generate_follow_up_questions, format_follow_up_questions
from safety_module import get_medical_attention_guidance, get_disclaimer, SafetyEscalationEvaluator
from analysis_module import MedicalSymptomAnalyzer, MockMedicalAnalyzer
from save_module import save_medical_report, ReportSaver, DEFAULT_OUTPUT_FILE
from input_module import load_sample_cases, get_profile_from_sample_case
from utils import DISCLAIMER_TEXT


# ------------------------------------------------------------------------------
# Part A – User Input (Q1)
# ------------------------------------------------------------------------------
def test_part_a_patient_profile():
    """Verify PatientProfile holds all required Q1 fields and formats properly."""
    profile = get_default_sample_profile()
    assert profile.age == 22
    assert profile.sex == "Female"
    assert profile.duration == "2 days"
    assert profile.severity == "Mild"
    assert profile.medical_history == "No known relevant condition"
    assert profile.is_complete()

    formatted = profile.format_sample_style()
    assert "Age : 22" in formatted
    assert "Sex : Female" in formatted
    assert "Duration : 2 days" in formatted
    assert "Severity : Mild" in formatted
    assert "History : No known relevant condition" in formatted


def test_part_a_profile_validation():
    """Verify validation detects invalid ages or severities."""
    invalid_profile = PatientProfile(age=250, severity="ExtremeDanger")
    is_valid, errors = invalid_profile.validate()
    assert not is_valid
    assert len(errors) >= 1


# ------------------------------------------------------------------------------
# Part B – Symptom Extraction (Q2)
# ------------------------------------------------------------------------------
def test_part_b_symptom_extraction_lab_sample():
    """
    Test extraction of symptoms from the exact example in Part B (Q2):
    User Input:
    "I have been having fever and cough for two days.
    My throat is also painful."
    Extracted Symptoms:
    • Fever
    • Cough
    • Sore throat
    """
    text = "I have been having fever and cough for two days. My throat is also painful."
    extracted = extract_symptoms_from_text(text)

    # Check case-insensitively
    extracted_lower = [s.lower() for s in extracted]
    assert "fever" in extracted_lower
    assert "cough" in extracted_lower
    assert "sore throat" in extracted_lower

    formatted = format_extracted_symptoms(extracted)
    assert "Extracted Symptoms:" in formatted
    assert "•" in formatted


# ------------------------------------------------------------------------------
# Part C – Context Analysis (Q3)
# ------------------------------------------------------------------------------
def test_part_c_context_analysis_missing_info():
    """
    Test context analysis identifies missing information matching Part C (Q3):
    Missing Information:
    • Exact temperature
    • Breathing difficulty
    • Severity progression
    • Additional symptoms
    """
    profile = get_default_sample_profile()
    missing = analyze_missing_context(profile)

    missing_str = " ".join(missing).lower()
    assert "temperature" in missing_str
    assert "breathing" in missing_str
    assert "progression" in missing_str
    assert "additional symptoms" in missing_str

    formatted = format_missing_information(missing)
    assert "Missing Information:" in formatted
    assert "• Exact temperature" in formatted
    assert "• Breathing difficulty" in formatted
    assert "• Severity progression" in formatted
    assert "• Additional symptoms" in formatted


# ------------------------------------------------------------------------------
# Part D – Prompt Generation (Q4)
# ------------------------------------------------------------------------------
def test_part_d_prompt_generation_constraints():
    """
    Verify constructed prompt satisfies all Part D (Q4) instructions:
    - Analyze symptoms
    - Identify possible conditions
    - Explain the reasoning at a general level
    - Ask follow-up questions
    - Identify when professional medical care may be appropriate
    - Avoid definitive diagnosis
    - Avoid inventing information
    """
    profile = get_default_sample_profile()
    prompt = construct_analysis_prompt(profile, profile.symptoms, ["Exact temperature"])

    # System prompt constraints
    sys_lower = SYSTEM_PROMPT.lower()
    assert "analyze symptoms" in sys_lower
    assert "possible conditions" in sys_lower
    assert "reasoning" in sys_lower
    assert "follow-up questions" in sys_lower
    assert "professional medical care" in sys_lower
    assert "avoid definitive diagnosis" in sys_lower
    assert "avoid inventing information" in sys_lower

    # User prompt constraints
    assert "Age: 22" in prompt
    assert "Sex: Female" in prompt
    assert "Fever, Cough, Sore throat" in prompt


# ------------------------------------------------------------------------------
# Part E – Symptom Analysis (Q5)
# ------------------------------------------------------------------------------
def test_part_e_preliminary_symptom_analysis():
    """
    Verify preliminary symptom analysis matches Part E (Q5) sample output:
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
    profile = get_default_sample_profile()
    analyzer = MedicalSymptomAnalyzer()
    output = analyzer.generate_preliminary_analysis(profile)

    assert "SYMPTOM ANALYSIS" in output
    assert "Reported Symptoms:" in output
    assert "• Fever" in output
    assert "• Cough" in output
    assert "• Sore throat" in output
    assert "Possible Conditions:" in output
    assert "Common cold" in output
    assert "Influenza" in output
    assert "Other respiratory infections" in output
    assert "Note:" in output
    assert "These symptoms can occur in multiple conditions." in output
    assert "The information provided is not sufficient to" in output
    assert "establish a diagnosis." in output


# ------------------------------------------------------------------------------
# Part F – Follow-Up Questions (Q6)
# ------------------------------------------------------------------------------
def test_part_f_follow_up_questions():
    """
    Verify generated follow-up questions match Part F (Q6) example:
    1. What is your measured temperature?
    2. When exactly did the symptoms begin?
    3. Are you experiencing difficulty breathing?
    4. Is the cough dry or producing mucus?
    5. Are the symptoms improving or worsening?
    """
    profile = get_default_sample_profile()
    questions = generate_follow_up_questions(profile)

    assert len(questions) >= 5
    assert "What is your measured temperature?" in questions
    assert "When exactly did the symptoms begin?" in questions
    assert "Are you experiencing difficulty breathing?" in questions
    assert "Is the cough dry or producing mucus?" in questions
    assert "Are the symptoms improving or worsening?" in questions

    formatted = format_follow_up_questions(questions)
    assert "FOLLOW-UP QUESTIONS" in formatted
    assert "1. What is your measured temperature?" in formatted


# ------------------------------------------------------------------------------
# Part G – Risk / Escalation Assessment (Q7)
# ------------------------------------------------------------------------------
def test_part_g_medical_attention_guidance_mild():
    """
    Verify medical attention guidance uses cautious language matching Part G (Q7):
    MEDICAL ATTENTION
    --------------------------------
    If symptoms are severe, rapidly worsening,
    or concerning, seek appropriate professional
    medical care.
    """
    profile = get_default_sample_profile()
    guidance = get_medical_attention_guidance(profile)

    assert "MEDICAL ATTENTION" in guidance
    assert "If symptoms are severe, rapidly worsening," in guidance
    assert "or concerning, seek appropriate professional" in guidance
    assert "medical care." in guidance


def test_part_g_medical_attention_guidance_urgent():
    """Verify red-flag symptoms trigger immediate urgent care alert."""
    urgent_profile = PatientProfile(
        age=60,
        symptoms=["Chest pain", "Shortness of breath"],
        raw_symptoms_text="Severe chest pressure radiating to arm",
        severity="Severe",
    )
    guidance = get_medical_attention_guidance(urgent_profile)
    assert "CRITICAL WARNING" in guidance or "urgent" in guidance.lower()


# ------------------------------------------------------------------------------
# Part H – Structured Response (Q8)
# ------------------------------------------------------------------------------
def test_part_h_structured_response_8_sections():
    """
    Verify complete structured response contains all 8 required sections in order:
    1. Reported Symptoms
    2. Patient Information
    3. Possible Conditions
    4. Explanation
    5. Missing Information
    6. Follow-up Questions
    7. Medical Attention Guidance
    8. Disclaimer
    """
    profile = get_default_sample_profile()
    analyzer = MedicalSymptomAnalyzer()
    report = analyzer.generate_structured_response(profile)

    assert "1. Reported Symptoms" in report
    assert "2. Patient Information" in report
    assert "3. Possible Conditions" in report
    assert "4. Explanation" in report
    assert "5. Missing Information" in report
    assert "6. Follow-up Questions" in report
    assert "7. Medical Attention Guidance" in report
    assert "8. Disclaimer" in report

    # Verify order of appearance
    pos1 = report.find("1. Reported Symptoms")
    pos2 = report.find("2. Patient Information")
    pos3 = report.find("3. Possible Conditions")
    pos4 = report.find("4. Explanation")
    pos5 = report.find("5. Missing Information")
    pos6 = report.find("6. Follow-up Questions")
    pos7 = report.find("7. Medical Attention Guidance")
    pos8 = report.find("8. Disclaimer")

    assert 0 <= pos1 < pos2 < pos3 < pos4 < pos5 < pos6 < pos7 < pos8


# ------------------------------------------------------------------------------
# Part I – Disclaimer (Q9)
# ------------------------------------------------------------------------------
def test_part_i_disclaimer_exact_text():
    """
    Verify exact disclaimer text matches Part I (Q9) specification:
    DISCLAIMER
    --------------------------------
    This AI-generated response is intended for
    educational and informational purposes only.
    It is not a medical diagnosis and does not
    replace advice from a qualified healthcare
    professional.
    For serious, worsening or urgent symptoms,
    seek appropriate medical care.
    """
    disclaimer = get_disclaimer()
    assert "DISCLAIMER" in disclaimer
    assert "educational and informational purposes only" in disclaimer
    assert "not a medical diagnosis" in disclaimer
    assert "replace advice from a qualified healthcare" in disclaimer
    assert "For serious, worsening or urgent symptoms," in disclaimer
    assert "seek appropriate medical care." in disclaimer


# ------------------------------------------------------------------------------
# Part J – Save Analysis (Q10)
# ------------------------------------------------------------------------------
def test_part_j_save_analysis():
    """Verify report can be saved to outputs/medical_analysis.txt and read back."""
    profile = get_default_sample_profile()
    analyzer = MedicalSymptomAnalyzer()
    report = analyzer.generate_structured_response(profile)

    target_file = os.path.join("outputs", "medical_analysis.txt")
    ok, msg = save_medical_report(report, file_path=target_file, append=False)
    assert ok
    assert os.path.exists(target_file)

    saved_text = ReportSaver.read_saved_analysis(target_file)
    assert saved_text is not None
    assert "1. Reported Symptoms" in saved_text
    assert "8. Disclaimer" in saved_text


# ------------------------------------------------------------------------------
# Part K – Complete Application (Q11) & Sample Cases
# ------------------------------------------------------------------------------
def test_sample_cases_dataset():
    """Verify sample_cases.csv is populated and can be loaded."""
    cases = load_sample_cases()
    assert len(cases) >= 5
    first_case = get_profile_from_sample_case(1)
    assert first_case is not None
    assert first_case.age == 22
    assert "Fever" in first_case.symptoms
