"""
Module: save_module.py
Experiment 6 – AI-Powered Medical Symptom Assistant using GPT-4
Part J – Save Analysis (Q10)
Saves structured symptom analysis reports to outputs/medical_analysis.txt.
"""

import os
from datetime import datetime
from typing import Tuple, Optional

# Default output file path as requested in Part J (Q10)
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_OUTPUT_FILE = os.path.join(DEFAULT_OUTPUT_DIR, "medical_analysis.txt")


class ReportSaver:
    """
    Manages persistence and file export for generated medical analyses.
    """

    @staticmethod
    def save_analysis(
        analysis_text: str,
        file_path: str = DEFAULT_OUTPUT_FILE,
        append: bool = False,
        include_timestamp: bool = True,
    ) -> Tuple[bool, str]:
        """
        Save analysis text to the designated output file.
        Returns (success_flag, status_message).
        """
        if not analysis_text or not analysis_text.strip():
            return False, "Cannot save an empty medical analysis."

        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(file_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            mode = "a" if append else "w"
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open(file_path, mode, encoding="utf-8") as f:
                if append:
                    f.write("\n\n" + "#" * 72 + "\n")
                    f.write(f"# REPORT GENERATED AT: {now_str}\n")
                    f.write("#" * 72 + "\n\n")
                elif include_timestamp:
                    header = f"# MEDICAL SYMPTOM ANALYSIS REPORT\n# Generated: {now_str}\n\n"
                    if not analysis_text.startswith("#"):
                        f.write(header)

                f.write(analysis_text.strip() + "\n")

            abs_path = os.path.abspath(file_path)
            return True, f"Analysis successfully saved to: {abs_path}"

        except Exception as e:
            return False, f"Failed to save analysis: {str(e)}"

    @staticmethod
    def read_saved_analysis(file_path: str = DEFAULT_OUTPUT_FILE) -> Optional[str]:
        """Read and return content of the saved analysis file."""
        if not os.path.exists(file_path):
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return None


def save_medical_report(
    analysis_text: str, file_path: str = DEFAULT_OUTPUT_FILE, append: bool = False
) -> Tuple[bool, str]:
    """Convenience helper to save medical report."""
    return ReportSaver.save_analysis(analysis_text, file_path=file_path, append=append)
