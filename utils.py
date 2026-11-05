"""
Module: utils.py
Experiment 6 – AI-Powered Medical Symptom Assistant using GPT-4
Helper utilities, formatting, clinical dictionaries, and disclaimers.
"""

import os
import re
from typing import List, Dict, Any, Optional

# ==============================================================================
# STANDARD MEDICAL DISCLAIMER (Part I – Q9)
# ==============================================================================
DISCLAIMER_TEXT = """DISCLAIMER
--------------------------------
This AI-generated response is intended for
educational and informational purposes only.
It is not a medical diagnosis and does not
replace advice from a qualified healthcare
professional.
For serious, worsening or urgent symptoms,
seek appropriate medical care."""

# Short disclaimer snippet for headers
SHORT_DISCLAIMER = "Informational only - Not a medical diagnosis. Consult a qualified physician."

# Clinical symptom taxonomy and pattern rules for NLP extraction fallback
COMMON_SYMPTOMS_MAP = {
    # Respiratory & ENT
    "fever": ["fever", "high temperature", "pyrexia", "feeling hot", "febrile", "burning up"],
    "cough": ["cough", "coughing", "dry cough", "productive cough", "hacking cough", "clearing throat"],
    "sore throat": [
        "sore throat", "throat is painful", "throat is also painful", "painful throat", 
        "scratchy throat", "throat pain", "hurts to swallow", "throat is sore", "throat hurts"
    ],
    "runny nose": ["runny nose", "rhinorrhea", "dripping nose", "nasal discharge"],
    "nasal congestion": ["stuffy nose", "blocked nose", "congested", "nasal congestion"],
    "sneezing": ["sneezing", "sneeze"],
    "shortness of breath": ["shortness of breath", "difficulty breathing", "dyspnea", "hard to breathe", "breathless", "can't catch breath", "wheezing"],
    "loss of smell": ["loss of smell", "anosmia", "cannot smell", "can't smell"],
    "loss of taste": ["loss of taste", "ageusia", "cannot taste", "can't taste"],
    "ear pain": ["earache", "ear pain", "pain in ear", "throbbing ear"],

    # Systemic & Musculoskeletal
    "fatigue": ["fatigue", "tiredness", "exhaustion", "lethargy", "feeling tired", "weakness"],
    "headache": ["headache", "head pain", "throbbing head", "migraine", "pain in head"],
    "body aches": ["body aches", "muscle aches", "myalgia", "body pain", "aching joints", "joint pain", "generalized aches"],
    "chills": ["chills", "shivering", "feeling cold", "rigors"],
    "dizziness": ["dizziness", "lightheadedness", "vertigo", "feeling faint", "room spinning"],

    # Gastrointestinal
    "nausea": ["nausea", "feeling sick", "queasy", "upset stomach"],
    "vomiting": ["vomiting", "threw up", "throwing up", "emesis"],
    "diarrhea": ["diarrhea", "loose stools", "watery stools", "upset bowels"],
    "abdominal pain": ["abdominal pain", "stomach pain", "stomach ache", "belly cramps", "cramping in stomach", "gut pain"],
    "loss of appetite": ["loss of appetite", "not hungry", "reduced appetite", "anorexia"],

    # Cardiovascular & Red Flags
    "chest pain": ["chest pain", "chest pressure", "tightness in chest", "heaviness in chest", "angina"],
    "palpitations": ["palpitations", "racing heart", "heart pounding", "irregular heartbeat"],

    # Dermatological
    "skin rash": ["rash", "skin eruption", "hives", "itchy bumps", "red spots"],
    "itching": ["itching", "pruritus", "itchy skin"],
}

# Red-flag severe symptoms requiring urgent escalation (Part G – Q7)
RED_FLAG_SYMPTOMS = [
    "chest pain",
    "shortness of breath",
    "difficulty breathing",
    "confusion",
    "loss of consciousness",
    "bluish lips",
    "coughing up blood",
    "severe abdominal pain",
    "sudden weakness",
    "slurred speech",
    "stiff neck with fever",
    "anaphylaxis",
]

# Severity levels
SEVERITY_LEVELS = ["Mild", "Moderate", "Severe", "Critical"]

# ==============================================================================
# TERMINAL COLORING & STYLING HELPERS
# ==============================================================================
class Colors:
    """ANSI color escapes for modern terminal rendering."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"

    # Foreground
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"

    # Background
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"


def print_banner(title: str, subtitle: Optional[str] = None):
    """Print an attractive application banner."""
    width = 72
    print("\n" + "=" * width)
    print(f" {title.center(width - 2)} ")
    if subtitle:
        print(f" {subtitle.center(width - 2)} ")
    print("=" * width)


def print_section(heading: str):
    """Print a clean section divider."""
    print(f"\n--- {heading.upper()} ---")


def format_bullet_list(items: List[str]) -> str:
    """Format a list as bullet points with clean bullet character."""
    if not items:
        return "• None noted"
    return "\n".join([f"• {item}" for item in items])


def format_numbered_list(items: List[str]) -> str:
    """Format a list as numbered items."""
    if not items:
        return "1. None noted"
    return "\n".join([f"{idx + 1}. {item}" for idx, item in enumerate(items)])


def clean_text(text: str) -> str:
    """Normalize whitespace and strip text."""
    if not text:
        return ""
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()
