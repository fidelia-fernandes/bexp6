"""
Module: symptom_module.py
Experiment 6 – AI-Powered Medical Symptom Assistant using GPT-4
Part B – Symptom Extraction (Q2)
Extracts standardized clinical symptoms from natural-language patient descriptions.
"""

import os
import re
import json
from typing import List, Optional
from dotenv import load_dotenv

from utils import COMMON_SYMPTOMS_MAP, clean_text, format_bullet_list

# Load environment variables if available
load_dotenv()


class SymptomExtractor:
    """
    Extracts clinical symptoms from unstructured natural language inputs.
    Supports both OpenAI GPT-4 live extraction and an offline rule-based NLP parser.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model

    def extract(self, user_text: str) -> List[str]:
        """
        Extract symptoms from natural language text.
        Attempts LLM extraction if API key is present; falls back to NLP heuristic parser.
        """
        if not user_text or not user_text.strip():
            return []

        if self.api_key and not self.api_key.startswith("mock_") and not self.api_key.startswith("sk-dummy"):
            try:
                llm_result = self._extract_via_llm(user_text)
                if llm_result:
                    return llm_result
            except Exception:
                # Silently fallback to NLP rule-based extraction
                pass

        return self._extract_heuristic(user_text)

    def _extract_via_llm(self, user_text: str) -> List[str]:
        """Extract symptoms using OpenAI GPT-4 API."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)

            system_prompt = (
                "You are an expert clinical NLP system. Your task is to extract all reported symptoms "
                "from the patient's free-text input. Standardize each symptom into concise clinical terminology "
                "(e.g., 'throat is painful' -> 'Sore throat', 'fever' -> 'Fever', 'cough' -> 'Cough').\n"
                "Respond ONLY with a valid JSON array of strings, for example: [\"Fever\", \"Cough\", \"Sore throat\"].\n"
                "Do not include diagnoses, duration, or explanations in this list."
            )

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                temperature=0.0,
                max_tokens=200,
            )

            raw_content = response.choices[0].message.content.strip()
            # Clean markdown code blocks if present
            raw_content = re.sub(r"^```(?:json)?\s*", "", raw_content)
            raw_content = re.sub(r"\s*```$", "", raw_content)
            parsed = json.loads(raw_content)

            if isinstance(parsed, list):
                # Title case and strip
                return [s.strip().capitalize() for s in parsed if isinstance(s, str) and s.strip()]
        except Exception:
            pass

        return []

    def _extract_heuristic(self, user_text: str) -> List[str]:
        """
        Robust offline rule-based NLP symptom extraction.
        Normalizes colloquial language, detects compound conjunctions,
        and standardizes terminology.
        """
        text = clean_text(user_text).lower()
        extracted: List[str] = []

        # 1. Direct synonym matching across symptom ontology
        for canonical_name, variants in COMMON_SYMPTOMS_MAP.items():
            for variant in variants:
                pattern = r"\b" + re.escape(variant) + r"\b"
                if re.search(pattern, text):
                    # Format as title or sentence case ("Sore throat")
                    formatted_name = canonical_name.capitalize()
                    if formatted_name not in extracted:
                        extracted.append(formatted_name)
                    break

        # Additional regex for flexible natural language phrases:
        # e.g., "throat is also painful", "throat is very sore", "throat really hurts"
        if re.search(r"\bthroat\b.*?\b(?:pain|painful|sore|hurts|scratchy)\b", text):
            if "Sore throat" not in extracted:
                extracted.append("Sore throat")

        # 2. Heuristic extraction of common patterns like "having fever and cough"
        # Match "having/had/experienced [symptom1], [symptom2] and [symptom3]"
        list_patterns = [
            r"(?:having|have|got|suffering from|experiencing)\s+([a-zA-Z\s,]+?)(?:\s+for|\s+since|\.|\n|$)",
            r"symptoms\s*(?:are|include|:)?\s*([a-zA-Z\s,]+?)(?:\.|\n|$)",
        ]
        for pat in list_patterns:
            match = re.search(pat, text)
            if match:
                raw_chunk = match.group(1)
                tokens = re.split(r",|\band\b", raw_chunk)
                for tok in tokens:
                    tok_clean = tok.strip()
                    # Filter out stopwords
                    if tok_clean and tok_clean not in ["a", "an", "the", "some", "severe", "mild"]:
                        for canon, vars_ in COMMON_SYMPTOMS_MAP.items():
                            if any(v in tok_clean for v in vars_):
                                c_title = canon.title()
                                if c_title not in extracted:
                                    extracted.append(c_title)

        # 3. Handle explicit comma-separated symptom lists (e.g. from structured input)
        if not extracted and "," in user_text:
            items = [item.strip() for item in re.split(r",|\band\b", user_text) if item.strip()]
            for item in items:
                cleaned = re.sub(r"^(?:having|mild|moderate|severe)\s+", "", item.strip(), flags=re.I)
                if cleaned:
                    extracted.append(cleaned.capitalize())

        # If still empty but text was provided, return cleaned snippet as single symptom
        if not extracted and user_text.strip():
            extracted.append(user_text.strip().capitalize())

        return extracted


def extract_symptoms_from_text(text: str, api_key: Optional[str] = None) -> List[str]:
    """Convenience function to extract symptoms from text."""
    extractor = SymptomExtractor(api_key=api_key)
    return extractor.extract(text)


def format_extracted_symptoms(symptoms: List[str]) -> str:
    """
    Format extracted symptoms matching Part B (Q2) specification:
    Extracted Symptoms:
    • Fever
    • Cough
    • Sore throat
    """
    bullets = format_bullet_list(symptoms)
    return f"Extracted Symptoms:\n{bullets}"
