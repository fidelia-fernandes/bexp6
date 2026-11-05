"""
Medical Symptom Assistant - Main Entrypoint
24CS2506 – Building Applications using GPT-4
Experiment 6 – AI-Powered Medical Symptom Assistant using GPT-4

Part K – Complete Application (Q11)
Integrates all modules into an interactive application offering:
1. Enter Patient Information
2. Enter Symptoms
3. Analyze Symptoms
4. Generate Follow-up Questions
5. Display Possible Conditions
6. Display Medical Attention Guidance
7. Display Disclaimer
8. Save Report
9. Exit
"""

import sys
import os
from typing import Optional

# Ensure current working directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from profile_module import PatientProfile, get_default_sample_profile
from input_module import PatientInputCollector, load_sample_cases, get_profile_from_sample_case
from symptom_module import extract_symptoms_from_text, format_extracted_symptoms
from context_module import analyze_missing_context, format_missing_information
from prompt_module import construct_analysis_prompt, SYSTEM_PROMPT
from question_module import generate_follow_up_questions, format_follow_up_questions
from safety_module import get_medical_attention_guidance, get_disclaimer, SafetyEscalationEvaluator
from analysis_module import MedicalSymptomAnalyzer
from save_module import save_medical_report, DEFAULT_OUTPUT_FILE
from utils import print_banner, DISCLAIMER_TEXT


class MedicalAssistantApp:
    """Integrated Console Application for Experiment 6."""

    def __init__(self):
        # Default with standard sample case from manual
        self.profile: PatientProfile = get_default_sample_profile()
        self.analyzer = MedicalSymptomAnalyzer()
        self.last_analysis_text: Optional[str] = None

    def display_app_header(self):
        """Display stylish top header showing active patient summary and engine status."""
        width = 72
        print("\n" + "=" * width)
        print("     AI-POWERED MEDICAL SYMPTOM ASSISTANT (POWERED BY GPT-4)      ".center(width))
        print("          24CS2506 - Building Applications using GPT-4            ".center(width))
        print("=" * width)
        print(f" Engine Status : {self.analyzer.get_mode_description()}")

        syms = ", ".join(self.profile.symptoms) if self.profile.symptoms else "None extracted"
        age_str = f"{self.profile.age}yo" if self.profile.age is not None else "N/A"
        print(f" Patient Active: Age: {age_str} | Sex: {self.profile.sex} | Severity: {self.profile.severity}")
        print(f" Symptoms      : {syms}")
        print("=" * width)

    def display_menu(self):
        """Display the 9-option interactive menu required by Part K (Q11)."""
        print("\nPlease select an option:")
        print("  1. Enter Patient Information")
        print("  2. Enter Symptoms")
        print("  3. Analyze Symptoms")
        print("  4. Generate Follow-up Questions")
        print("  5. Display Possible Conditions")
        print("  6. Display Medical Attention Guidance")
        print("  7. Display Disclaimer")
        print("  8. Save Report")
        print("  9. Exit")
        print("  ──────────────────────────────────────────")
        print("  0. Load Preset Sample Case (from CSV)")
        print("-" * 72)

    # --------------------------------------------------------------------------
    # Option 1: Enter Patient Information (Part A – Q1)
    # --------------------------------------------------------------------------
    def handle_enter_patient_info(self):
        """1. Enter Patient Information."""
        print("\n[>>] Option 1: Enter Patient Information")
        self.profile = PatientInputCollector.collect_patient_profile(existing=self.profile)
        print("\nCURRENT PATIENT INFORMATION (Sample Input Format):")
        print("-" * 50)
        print(self.profile.format_sample_style())
        print("-" * 50)
        # Reset cached full report on change
        self.last_analysis_text = None

    # --------------------------------------------------------------------------
    # Option 2: Enter Symptoms (Part B – Q2)
    # --------------------------------------------------------------------------
    def handle_enter_symptoms(self):
        """2. Enter Symptoms & Run Natural-Language Extraction."""
        print("\n[>>] Option 2: Enter Symptoms")
        narrative = PatientInputCollector.collect_symptoms_narrative(existing=self.profile)
        self.profile.raw_symptoms_text = narrative

        print("\n[*] Extracting symptoms using Clinical NLP / GPT-4...")
        extracted = self.analyzer.symptom_extractor.extract(narrative)
        self.profile.symptoms = extracted

        print("\n" + "=" * 50)
        print(format_extracted_symptoms(extracted))
        print("=" * 50)
        self.last_analysis_text = None

    # --------------------------------------------------------------------------
    # Option 3: Analyze Symptoms (Part E – Q5 & Part H – Q8)
    # --------------------------------------------------------------------------
    def handle_analyze_symptoms(self):
        """3. Analyze Symptoms & Generate Complete Structured Response."""
        print("\n[>>] Option 3: Analyze Symptoms")
        print("[*] Generating AI Clinical Analysis (analyzing symptoms, context, differential)...")

        # Generate full structured report (Part H – Q8)
        self.last_analysis_text = self.analyzer.generate_structured_response(self.profile)

        print("\n" + self.last_analysis_text)
        print("\n[i] Structured analysis is ready. You can save it via Option 8.")

    # --------------------------------------------------------------------------
    # Option 4: Generate Follow-up Questions (Part F – Q6)
    # --------------------------------------------------------------------------
    def handle_generate_follow_up_questions(self):
        """4. Generate Follow-up Questions."""
        print("\n[>>] Option 4: Generate Follow-up Questions")
        missing = analyze_missing_context(self.profile)
        questions = generate_follow_up_questions(self.profile, missing)

        print("\n" + "=" * 50)
        print(format_follow_up_questions(questions))
        print("=" * 50)

    # --------------------------------------------------------------------------
    # Option 5: Display Possible Conditions (Part E – Q5)
    # --------------------------------------------------------------------------
    def handle_display_possible_conditions(self):
        """5. Display Possible Conditions (Preliminary Symptom Analysis)."""
        print("\n[>>] Option 5: Display Possible Conditions")
        preliminary_output = self.analyzer.generate_preliminary_analysis(self.profile)

        print("\n" + "=" * 50)
        print(preliminary_output)
        print("=" * 50)

    # --------------------------------------------------------------------------
    # Option 6: Display Medical Attention Guidance (Part G – Q7)
    # --------------------------------------------------------------------------
    def handle_display_medical_attention_guidance(self):
        """6. Display Medical Attention Guidance."""
        print("\n[>>] Option 6: Display Medical Attention Guidance")
        guidance = get_medical_attention_guidance(self.profile)

        print("\n" + "=" * 50)
        print(guidance)
        print("=" * 50)

    # --------------------------------------------------------------------------
    # Option 7: Display Disclaimer (Part I – Q9)
    # --------------------------------------------------------------------------
    def handle_display_disclaimer(self):
        """7. Display Disclaimer."""
        print("\n[>>] Option 7: Display Disclaimer")
        print("\n" + "=" * 50)
        print(get_disclaimer())
        print("=" * 50)

    # --------------------------------------------------------------------------
    # Option 8: Save Report (Part J – Q10)
    # --------------------------------------------------------------------------
    def handle_save_report(self):
        """8. Save Report to outputs/medical_analysis.txt."""
        print("\n[>>] Option 8: Save Report")
        if not self.last_analysis_text:
            print("[*] Generating full structured analysis before saving...")
            self.last_analysis_text = self.analyzer.generate_structured_response(self.profile)

        append_choice = input("Append to existing file? (y/N) [default: Overwrite latest]: ").strip().lower()
        append = append_choice == "y"

        success, msg = save_medical_report(self.last_analysis_text, append=append)
        if success:
            print(f"\n[OK] {msg}")
        else:
            print(f"\n[!] {msg}")

    # --------------------------------------------------------------------------
    # Option 0: Load Preset Sample Case
    # --------------------------------------------------------------------------
    def handle_load_sample_case(self):
        """0. Select and load preset sample case."""
        preset = PatientInputCollector.display_sample_cases_menu()
        if preset:
            self.profile = preset
            self.last_analysis_text = None

    # --------------------------------------------------------------------------
    # Automated Demo / Self-Test Runner
    # --------------------------------------------------------------------------
    def run_automated_demo(self):
        """Run an automated demo demonstrating every module and question end-to-end."""
        print_banner(
            "AUTOMATED DEMONSTRATION MODE - EXPERIMENT 6",
            "Validating Parts A through K step-by-step",
        )

        print("\n" + "#" * 60)
        print("STEP 1: PATIENT INFORMATION (Part A – Q1)")
        print("#" * 60)
        print(self.profile.format_sample_style())

        print("\n" + "#" * 60)
        print("STEP 2: SYMPTOM EXTRACTION (Part B – Q2)")
        print("#" * 60)
        print(f"Raw Input: \"{self.profile.raw_symptoms_text}\"")
        extracted = self.analyzer.symptom_extractor.extract(self.profile.raw_symptoms_text)
        print(format_extracted_symptoms(extracted))

        print("\n" + "#" * 60)
        print("STEP 3: CONTEXT ANALYSIS / MISSING INFO (Part C – Q3)")
        print("#" * 60)
        missing = analyze_missing_context(self.profile)
        print(format_missing_information(missing))

        print("\n" + "#" * 60)
        print("STEP 4: GPT-4 PROMPT GENERATION (Part D – Q4)")
        print("#" * 60)
        prompt = construct_analysis_prompt(self.profile, extracted, missing)
        print(prompt[:400] + "\n... [Prompt truncated for display] ...")

        print("\n" + "#" * 60)
        print("STEP 5: SYMPTOM ANALYSIS & POSSIBLE CONDITIONS (Part E – Q5)")
        print("#" * 60)
        preliminary = self.analyzer.generate_preliminary_analysis(self.profile)
        print(preliminary)

        print("\n" + "#" * 60)
        print("STEP 6: FOLLOW-UP QUESTIONS (Part F – Q6)")
        print("#" * 60)
        questions = generate_follow_up_questions(self.profile, missing)
        print(format_follow_up_questions(questions))

        print("\n" + "#" * 60)
        print("STEP 7: MEDICAL ATTENTION GUIDANCE (Part G – Q7)")
        print("#" * 60)
        print(get_medical_attention_guidance(self.profile))

        print("\n" + "#" * 60)
        print("STEP 8: COMPLETE STRUCTURED RESPONSE (Part H – Q8)")
        print("#" * 60)
        structured_report = self.analyzer.generate_structured_response(self.profile)
        print(structured_report)

        print("\n" + "#" * 60)
        print("STEP 9: DISCLAIMER (Part I – Q9)")
        print("#" * 60)
        print(get_disclaimer())

        print("\n" + "#" * 60)
        print("STEP 10: SAVE TO outputs/medical_analysis.txt (Part J – Q10)")
        print("#" * 60)
        ok, msg = save_medical_report(structured_report, append=False)
        print(f"[OK] {msg}")

        print("\n" + "=" * 60)
        print("      ALL EXPERIMENT 6 MODULES VALIDATED SUCCESSFULLY!      ")
        print("=" * 60)

    # --------------------------------------------------------------------------
    # Main Application Loop (Part K – Q11)
    # --------------------------------------------------------------------------
    def run(self):
        """Interactive application main loop."""
        while True:
            self.display_app_header()
            self.display_menu()

            choice = input("\nEnter choice (1-9, or 0): ").strip()

            if choice == "1":
                self.handle_enter_patient_info()
            elif choice == "2":
                self.handle_enter_symptoms()
            elif choice == "3":
                self.handle_analyze_symptoms()
            elif choice == "4":
                self.handle_generate_follow_up_questions()
            elif choice == "5":
                self.handle_display_possible_conditions()
            elif choice == "6":
                self.handle_display_medical_attention_guidance()
            elif choice == "7":
                self.handle_display_disclaimer()
            elif choice == "8":
                self.handle_save_report()
            elif choice == "9":
                print("\n" + "=" * 60)
                print("Thank you for using the AI-Powered Medical Symptom Assistant.")
                print("Stay safe and take care of your health!")
                print("=" * 60)
                sys.exit(0)
            elif choice == "0":
                self.handle_load_sample_case()
            else:
                print("\n[!] Invalid selection. Please enter a number between 1 and 9.")

            input("\nPress Enter to return to main menu...")


def main():
    """Application CLI entrypoint."""
    app = MedicalAssistantApp()

    if "--demo" in sys.argv or "--sample" in sys.argv or "--test" in sys.argv:
        app.run_automated_demo()
    elif "--gui" in sys.argv or "--web" in sys.argv:
        # Launch Streamlit web dashboard
        import subprocess
        streamlit_script = os.path.join(os.path.dirname(__file__), "app_streamlit.py")
        subprocess.run(["streamlit", "run", streamlit_script])
    else:
        app.run()


if __name__ == "__main__":
    main()
