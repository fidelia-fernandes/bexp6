# 24CS2506 – Building Applications using GPT-4
## Experiment 6: AI-Powered Medical Symptom Assistant using GPT-4

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI GPT-4](https://img.shields.io/badge/OpenAI-GPT--4-orange.svg)](https://platform.openai.com/)
[![Tests](https://img.shields.io/badge/pytest-13%20passed-success.svg)](https://docs.pytest.org/)
[![UI](https://img.shields.io/badge/Streamlit-SaaS%20Dashboard-red.svg)](https://streamlit.io/)

---

## 📌 Problem Statement

Design and develop an **AI-powered Medical Symptom Assistant** using **GPT-4**. The application accepts user-provided health information and symptoms, analyzes the information using GPT-4, identifies possible conditions associated with the reported symptoms, generates relevant follow-up questions, and provides appropriate guidance regarding professional medical attention. 

The system strictly adheres to ethical medical safety guidelines: it communicates that its output is **strictly informational and educational**, never provides a definitive diagnosis, and avoids hallucinating medical details.

---

## 🏗️ Project Architecture & Structure

```
MedicalAssistant/ (bexp6/)
│
├── main.py                  # Integrated CLI Application (Part K - Q11) with 9-item menu
├── app_streamlit.py         # Modern interactive SaaS Web Dashboard (Streamlit)
│
├── profile_module.py        # Part A (Q1): Patient demographic & health profile data model
├── symptom_module.py        # Part B (Q2): Clinical NLP & GPT-4 natural-language symptom extraction
├── context_module.py        # Part C (Q3): Context analysis & missing clinical information detection
├── prompt_module.py         # Part D (Q4): Constrained prompt construction with safety guardrails
├── analysis_module.py       # Part E (Q5) & Part H (Q8): Symptom analysis & structured response
├── question_module.py       # Part F (Q6): Dynamic clinical follow-up question generator
├── safety_module.py         # Part G (Q7) & Part I (Q9): Risk escalation assessment & disclaimer
├── save_module.py           # Part J (Q10): File persistence to outputs/medical_analysis.txt
├── utils.py                 # Clinical taxonomy, formatting helpers, and standard disclaimer text
│
├── data/
│   └── sample_cases.csv     # Preset clinical cases (URTI, Gastroenteritis, Rhinitis, Red Flags)
│
├── outputs/
│   └── medical_analysis.txt # Generated structured analysis reports
│
├── tests/
│   └── test_medical_assistant.py # Comprehensive test suite (100% pass rate)
│
├── requirements.txt         # Project dependencies
├── .env.example             # Environment variable template
└── README.md                # Comprehensive documentation
```

---

## 🔬 Implementation Details (Parts A – K)

### Part A – User Input (Q1)
- Collects: Age, Sex, Symptoms, Duration, Severity, Medical history, Current medications, Allergies.
- Managed by [`profile_module.py`](file:///c:/Users/fidel/Downloads/bexp6/profile_module.py) and [`input_module.py`](file:///c:/Users/fidel/Downloads/bexp6/input_module.py).
- **Sample Input Format:**
  ```text
  Age : 22
  Sex : Female
  Symptoms : Fever, cough and sore throat
  Duration : 2 days
  Severity : Mild
  History : No known relevant condition
  ```

### Part B – Symptom Extraction (Q2)
- Extracts clinical symptoms from unstructured natural-language patient descriptions.
- Managed by [`symptom_module.py`](file:///c:/Users/fidel/Downloads/bexp6/symptom_module.py).
- **Example:**
  - *User Input:* `"I have been having fever and cough for two days. My throat is also painful."`
  - *Extracted Symptoms:*
    - `• Fever`
    - `• Cough`
    - `• Sore throat`

### Part C – Context Analysis (Q3)
- Identifies missing relevant clinical parameters (e.g., objective fever measurement, respiratory status, progression).
- Managed by [`context_module.py`](file:///c:/Users/fidel/Downloads/bexp6/context_module.py).
- **Example:**
  ```text
  Missing Information:
  • Exact temperature
  • Breathing difficulty
  • Severity progression
  • Additional symptoms
  ```

### Part D – Prompt Generation (Q4)
- Constructs safety-constrained prompts instructing GPT-4 to:
  1. Analyze symptoms
  2. Identify possible conditions
  3. Explain reasoning at a general level
  4. Ask follow-up questions
  5. Identify when professional medical care is appropriate
  6. Avoid definitive diagnosis
  7. Avoid inventing information
- Managed by [`prompt_module.py`](file:///c:/Users/fidel/Downloads/bexp6/prompt_module.py).

### Part E – Symptom Analysis (Q5)
- Uses GPT-4 (with intelligent offline clinical knowledge fallback) to formulate a preliminary analysis.
- Managed by [`analysis_module.py`](file:///c:/Users/fidel/Downloads/bexp6/analysis_module.py).
- **Sample Output:**
  ```text
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
  ```

### Part F – Follow-Up Questions (Q6)
- Synthesizes targeted clinical clarifying questions.
- Managed by [`question_module.py`](file:///c:/Users/fidel/Downloads/bexp6/question_module.py).
- **Example Output:**
  ```text
  FOLLOW-UP QUESTIONS
  --------------------------------
  1. What is your measured temperature?
  2. When exactly did the symptoms begin?
  3. Are you experiencing difficulty breathing?
  4. Is the cough dry or producing mucus?
  5. Are the symptoms improving or worsening?
  ```

### Part G – Risk / Escalation Assessment (Q7)
- Evaluates symptom severity and red-flag indicators (e.g., chest pain, shortness of breath, high persistent fever) with cautious language.
- Managed by [`safety_module.py`](file:///c:/Users/fidel/Downloads/bexp6/safety_module.py).
- **Example:**
  ```text
  MEDICAL ATTENTION
  --------------------------------
  If symptoms are severe, rapidly worsening,
  or concerning, seek appropriate professional
  medical care.
  ```

### Part H – Structured Response (Q8)
- Generates the complete 8-section report in the exact required schema:
  1. Reported Symptoms
  2. Patient Information
  3. Possible Conditions
  4. Explanation
  5. Missing Information
  6. Follow-up Questions
  7. Medical Attention Guidance
  8. Disclaimer

### Part I – Disclaimer (Q9)
- Displays the mandatory medical disclaimer:
  ```text
  DISCLAIMER
  --------------------------------
  This AI-generated response is intended for
  educational and informational purposes only.
  It is not a medical diagnosis and does not
  replace advice from a qualified healthcare
  professional.
  For serious, worsening or urgent symptoms,
  seek appropriate medical care.
  ```

### Part J – Save Analysis (Q10)
- Saves the complete structured report to:
  `outputs/medical_analysis.txt`
- Managed by [`save_module.py`](file:///c:/Users/fidel/Downloads/bexp6/save_module.py).

### Part K – Complete Application (Q11)
- Integrates all components into an interactive application providing the exact 9-option menu:
  ```text
  1. Enter Patient Information
  2. Enter Symptoms
  3. Analyze Symptoms
  4. Generate Follow-up Questions
  5. Display Possible Conditions
  6. Display Medical Attention Guidance
  7. Display Disclaimer
  8. Save Report
  9. Exit
  ```

---

## 🚀 Getting Started & Execution

### 1. Environment Setup
```bash
# Clone or navigate to the directory
cd c:\Users\fidel\Downloads\bexp6

# (Optional) Set up OpenAI API Key in .env
copy .env.example .env
# Edit .env and enter your key: OPENAI_API_KEY=sk-...
```

> **Note:** The assistant features a **dual-engine architecture**. If an OpenAI API key is provided, it calls GPT-4 (`gpt-4o`). If no key is set, it seamlessly activates the built-in clinical knowledge engine, guaranteeing 100% offline usability.

---

### 2. Run Interactive CLI Application (Part K)
```bash
python main.py
```

### 3. Run Automated End-to-End Demonstration Mode
To see all 11 parts execute automatically with the lab manual sample input:
```bash
python main.py --demo
```

### 4. Run Modern Streamlit SaaS Web Dashboard
```bash
streamlit run app_streamlit.py
```

---

## 🧪 Automated Test Suite

Run the full pytest suite:
```bash
python -m pytest -v tests/
```

**Results:**
```text
tests/test_medical_assistant.py::test_part_a_patient_profile PASSED      [  7%]
tests/test_medical_assistant.py::test_part_a_profile_validation PASSED   [ 15%]
tests/test_medical_assistant.py::test_part_b_symptom_extraction_lab_sample PASSED [ 23%]
tests/test_medical_assistant.py::test_part_c_context_analysis_missing_info PASSED [ 30%]
tests/test_medical_assistant.py::test_part_d_prompt_generation_constraints PASSED [ 38%]
tests/test_medical_assistant.py::test_part_e_preliminary_symptom_analysis PASSED [ 46%]
tests/test_medical_assistant.py::test_part_f_follow_up_questions PASSED  [ 53%]
tests/test_medical_assistant.py::test_part_g_medical_attention_guidance_mild PASSED [ 61%]
tests/test_medical_assistant.py::test_part_g_medical_attention_guidance_urgent PASSED [ 69%]
tests/test_medical_assistant.py::test_part_h_structured_response_8_sections PASSED [ 76%]
tests/test_medical_assistant.py::test_part_i_disclaimer_exact_text PASSED [ 84%]
tests/test_medical_assistant.py::test_part_j_save_analysis PASSED        [ 92%]
tests/test_medical_assistant.py::test_sample_cases_dataset PASSED        [100%]

============================= 13 passed in 0.19s ==============================
```

---

## 📄 License & Academic Integrity
Developed for **24CS2506 – Building Applications using GPT-4**. Intended strictly for educational demonstration.
