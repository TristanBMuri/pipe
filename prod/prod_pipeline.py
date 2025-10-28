import logging
import json
from dataclasses import asdict
import dotenv
from pathlib import Path

# --- Import functions from your provided scripts ---
# Assuming scripts are in the same directory or a package named 'experimental'
# We will use the correct filenames you uploaded.

# From experimental/ner_openai.py
from helpers.ner_openai import extract_medical_entities, ner_entity_comparison
# From experimental/nli_verification_ollama.py
import helpers.nli_langfuse as nli_langfuse
# From experimental/nli_verification.py (for data structures)
import helpers.nli_dataclasses as nli_hf

# --- 2. Thresholds Configuration (as requested) ---
NER_PASS_THRESHOLD = 50.0 

# NLI Thresholds (all set to 0.5 as requested)
NLI_THRESHOLDS = nli_hf.Thresholds(
    entailment_support=0.5,
    contradiction_support=0.5,  # Max contradiction allowed for a "supported" sentence
    max_sim_coverage=0.5,  # Min similarity for a chunk to be "covered"
    top_k_retrieval=3  # Using the value from your pipeline.py
)

# NLI Pass Criteria (all set to 0.5 as requested)
NLI_CRITERIA = nli_hf.PassCriteria(
    min_supported_rate=0.5,
    max_hallucination_rate=0.5,
    min_coverage=0.5,
    zero_contradictions=False  # Allows contradictions if they don't fail other criteria
)


# --- 3. Revision Prompt Generation ---

def generate_revision_prompt(input_check: str, ner_results: dict, nli_report: nli_hf.VerificationReport) -> str:
    """
    Generates a detailed prompt to "upgrade" the text based on NER and NLI failures.
    """
    prompt_parts = [
        "Please revise the following 'Text to Verify'.",
        "It was compared against a 'Ground Truth' text and found to have the following potential issues.",
        "Use the provided feedback to create an improved version that is more faithful to the Ground Truth where it makes sense.",
        "Medical terms that are present don't have to be overwritten by their non-medical counterparts."
        "The feedback is automatically generated and should not be blindly followed, but just be a consideration for potential improvement."
        "\n--- TEXT TO VERIFY ---",
        f'"{input_check}"',
        "\n--- FEEDBACK & ISSUES TO CORRECT ---"
    ]

    # 1. Add NER Feedback
    if ner_results.get("only_in_truth"):
        prompt_parts.append("\n[Missing Entities (May be added)]:")
        for ent in ner_results["only_in_truth"]:
            prompt_parts.append(f"- {ent.get('text')} ({ent.get('label')})")

    if ner_results.get("only_in_check"):
        prompt_parts.append("\n[Extra/Incorrect Entities (May be removed or corrected)]:")
        for ent in ner_results["only_in_check"]:
            prompt_parts.append(f"- {ent.get('text')} ({ent.get('label')})")

    # 2. Add NLI Feedback (Unsupported Sentences)
    if nli_report.unsupported:
        prompt_parts.append("\n[Unsupported or Contradictory Sentences (Must be revised for factual accuracy)]:")
        for sent_result in nli_report.unsupported:
            prompt_parts.append(f"- Sentence: \"{sent_result.sentence}\"")
            prompt_parts.append(f"  - Reason: {sent_result.note}")
            if sent_result.conflicting_evidence and sent_result.contradiction > nli_report.thresholds.contradiction_support:
                prompt_parts.append(f"  - Contradicted by: \"{sent_result.conflicting_evidence}\"")
            elif sent_result.supporting_evidence:
                prompt_parts.append(f"  - Best match: \"{sent_result.supporting_evidence}\"")

    prompt_parts.append("\n--- END OF FEEDBACK ---")
    prompt_parts.append("\nPlease provide only the revised text:")

    return "\n".join(prompt_parts)


# --- 4. Main Verification Pipeline ---

def run_verification_pipeline(input_truth: str, input_check: str, return_prompt=False) -> dict:
    """
    Runs the full NER and NLI verification pipeline.

    Returns:
        A dictionary containing the full analysis and a
        revision_prompt if verification fails.
    """

    # Start NER
    try:
        entities_truth = extract_medical_entities(input_truth)
        logging.info(f"Extracted {len(entities_truth)} entities from TRUTH")

        entities_check = extract_medical_entities(input_check)
        logging.info(f"Extracted {len(entities_check)} entities from CHECK")
    except Exception as e:
        logging.error(f"Error: {e}")
        return {"error": f"NER request failed: {e}"}

    percentage, only_in_truth, only_in_check = ner_entity_comparison(entities_truth, entities_check)

    ner_results = {
        "percentage": percentage,
        "only_in_truth": only_in_truth,
        "only_in_check": only_in_check,
        "passed": percentage >= NER_PASS_THRESHOLD
    }
    logging.info(f"NER Comparison: {percentage:.2f}% match. Passed: {ner_results['passed']}\n")

    logging.info("--- 2. Starting NLI Verification ---")

    try:
        nli_report = nli_langfuse.verify_draft_against_transcript(
            transcript_text=input_truth,
            draft_text=input_check,
            language="german",
            thresholds=NLI_THRESHOLDS,
            criteria=NLI_CRITERIA
        )
        logging.info(f"NLI Verification Passed: {nli_report.passed}\n")

    except Exception as e:
        logging.error(f"FATAL: NLI verification failed.")
        logging.error(f"Error: {e}")
        return {"error": f"Ollama NLI verification failed: {e}"}

    # --- 3. Final Verdict & Revision Prompt ---

    final_verdict = "Pass"
    revision_prompt = None

    if not ner_results["passed"] or not nli_report.passed:
        final_verdict = "Fail"
        logging.info("--- 4. Generating Revision Prompt ---")

        # Add thresholds to report for context in prompt generation
        nli_report.thresholds = NLI_THRESHOLDS

        revision_prompt = generate_revision_prompt(
            input_check=input_check,
            ner_results=ner_results,
            nli_report=nli_report
        )
        logging.info("Revision prompt generated.")

    return {
        "final_verdict": final_verdict,
        "revision_prompt": revision_prompt,
        "ner_results": ner_results,
        "nli_report": asdict(nli_report)  # Convert dataclass to dict for easy printing/JSON
    }


# --- 5. Example Usage ---
if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    ENV_PATH = PROJECT_ROOT / ".env"

    if ENV_PATH.exists():
        dotenv.load_dotenv(dotenv_path=ENV_PATH)
    else:
        print(f"Warning: .env file not found at {ENV_PATH}. Trying default load.")
        dotenv.load_dotenv()

    truth_text = (
        "Der Patient, Herr Schmidt, 62 Jahre alt, berichtet über Schwindel und "
        "plötzliche Kopfschmerzen seit gestern Abend. Er nimmt aktuell Ramipril 5mg "
        "gegen seinen Bluthochdruck. Eine bekannte Allergie gegen Penicillin besteht. "
        "Wir haben ein CT des Kopfes angeordnet, um eine Blutung auszuschließen."
    )

    check_text_pass = (
        "Herr Schmidt (62) hat Schwindel und Kopfschmerzen. Er nimmt Ramipril 5mg "
        "wegen Bluthochdruck. Er ist allergisch gegen Penicillin. Ein Kopf-CT wurde "
        "angeordnet."
    )

    check_text_fail = (
        "Herr Schmidt (62) hat Ohrenschmerzen. Er nimmt Ibuprofen 400mg "
        "gegen den Schmerz. Eine Allergie gegen Aspirin wurde vermerkt. "
        "Wir haben eine Röntgenaufnahme des Thorax gemacht."
    )

    print("===========================================")
    print("     RUNNING VERIFICATION (FAIL CASE)    ")
    print("===========================================")

    # Run the pipeline with the failing text
    results_fail = run_verification_pipeline(truth_text, check_text_fail)

    print("\n--- FINAL RESULTS (FAIL) ---")
    print(f"Verdict: {results_fail['final_verdict']}")
    print(f"NER Passed: {results_fail['ner_results']['passed']}")
    print(f"NLI Passed: {results_fail['nli_report']['passed']}")

    if results_fail["revision_prompt"]:
        print("\n--- Generated Revision Prompt ---")
        print(results_fail["revision_prompt"])

    print("\n\n===========================================")
    print("     RUNNING VERIFICATION (PASS CASE)    ")
    print("===========================================")

    # Run the pipeline with the passing text
    results_pass = run_verification_pipeline(truth_text, check_text_pass)

    print("\n--- FINAL RESULTS (PASS) ---")
    print(f"Verdict: {results_pass['final_verdict']}")
    print(f"NER Passed: {results_pass['ner_results']['passed']}")
    print(f"NLI Passed: {results_pass['nli_report']['passed']}")

    if results_pass["revision_prompt"]:
        print("\n--- Generated Revision Prompt ---")
        print(results_pass["revision_prompt"])
    else:
        print("\nNo revision prompt needed.")