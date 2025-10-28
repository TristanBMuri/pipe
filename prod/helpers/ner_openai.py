import json

import dotenv
from langfuse._client.observe import observe

from .LangfusePrompter_v3 import LangfusePrompter

@observe(name="ner", as_type="generation")
def extract_medical_entities(text):
    """
    Extracts medical entities from text using a specified Ollama model
    and a custom prompt.

    Entities to extract:
    - DISEASE
    - SYMPTOM
    - MEDICATION
    - PROCEDURE
    """
    response = "Not received yet"

    try:
        prompter = LangfusePrompter("medical_named_entity_extraction", "production")

        response = prompter.text_prompt("Nothing", ner_text=text)

        # The response content should be a JSON string
        content = response.content

        # Parse the JSON string
        data = json.loads(content)

        # Return the list of entities, or an empty list if key isn't found
        return data.get("entities", [])

    except json.JSONDecodeError:
        print("Error: Failed to decode JSON response.")
        print("Raw response:", response)
        return []
    except Exception as e:
        # Handle cases where Ollama service isn't running
        print(f"An error occurred: {e}")
        return []

def ner_entity_comparison(ner_list1, ner_list2):
    """
    Compares two lists of NER entity dictionaries.

    The comparison is case-insensitive on 'text' and 'label'.

    Args:
        ner_list1 (list): The first list of entities (e.g., ground truth).
        ner_list2 (list): The second list of entities (e.g., model output).

    Returns:
        tuple: (coverage_percentage, only_in_list1, only_in_list2)
            - coverage_percentage (float): Pct of unique entities from list1
                                           found in list2.
            - only_in_list1 (list): Full entity dicts present only in list1.
            - only_in_list2 (list): Full entity dicts present only in list2.
    """

    def _normalize_entity(e):
        # Creates a comparable tuple, ignoring case
        return (
            e.get('text', '').lower(),
            e.get('label', '').lower()
        )

    # Use sets for efficient comparison of unique entities
    set1_normalized = {_normalize_entity(e) for e in ner_list1}
    set2_normalized = {_normalize_entity(e) for e in ner_list2}

    # --- 1. Calculate Coverage ---
    common_entities = set1_normalized.intersection(set2_normalized)

    if not set1_normalized:
        # If list 1 is empty, coverage is 100% (all 0 entities were found)
        coverage_percentage = 100.0
    else:
        coverage_percentage = (len(common_entities) / len(set1_normalized)) * 100

    # --- 2. Find entities only in list 1 ---
    only_in_set1 = set1_normalized - set2_normalized
    only_in_list1 = []
    seen_in_only1 = set()  # Avoid adding duplicates from the original list

    for e in ner_list1:
        normalized_e = _normalize_entity(e)
        if normalized_e in only_in_set1 and normalized_e not in seen_in_only1:
            only_in_list1.append(e)
            seen_in_only1.add(normalized_e)

    # --- 3. Find entities only in list 2 ---
    only_in_set2 = set2_normalized - set1_normalized
    only_in_list2 = []
    seen_in_only2 = set()  # Avoid adding duplicates

    for e in ner_list2:
        normalized_e = _normalize_entity(e)
        if normalized_e in only_in_set2 and normalized_e not in seen_in_only2:
            only_in_list2.append(e)
            seen_in_only2.add(normalized_e)

    return coverage_percentage, only_in_list1, only_in_list2

if __name__ == "__main__":
    dotenv.load_dotenv()
    example_text = (
        "The patient complains of severe headache and dizziness. "
        "Aspirin was prescribed for pain relief. "
        "An MRI scan is scheduled to rule out a brain tumor. "
        "He has a history of hypertension and diabetes."
    )
    file_path = "test_transcription_copy.txt"

    with open(file_path, "r", encoding="utf-8") as f:
        example_text = f.read()

    re= extract_medical_entities(example_text)

    print(re)