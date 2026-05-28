import argparse
import json
import os
import re
import sys

import numpy as np
from open_webui_client import call_model


TOKEN_LIMIT_KEYWORDS = [
    "token",
    "context",
    "length",
    "too long",
    "maximum",
    "prompt is too long",
]

DEFAULT_MAX_PATIENT_JSON_CHARS = 120_000


def load_json_file(path):
    with open(path, "r") as f:
        return json.load(f)


def truncate_text(text, max_chars=DEFAULT_MAX_PATIENT_JSON_CHARS):
    if max_chars is None or len(text) <= max_chars:
        return text

    print(f"  ⚠ Patient JSON too long ({len(text)} chars). Truncating to {max_chars} chars.")
    return (
        text[:max_chars]
        + "\n\n[TRUNCATED: patient JSON exceeded maximum allowed length]"
    )


def prepare_patient_data(original_patient_data, trim_data=False, max_patient_json_chars=DEFAULT_MAX_PATIENT_JSON_CHARS):
    if trim_data:
        patient_data = trim_patient_data(original_patient_data)
    else:
        patient_data = original_patient_data

    patient_data_json = json.dumps(patient_data, indent=2)
    patient_data_json = truncate_text(patient_data_json, max_patient_json_chars)

    return patient_data, patient_data_json


def extract_error_message(error):
    error_msg = str(error).lower()

    if hasattr(error, "response") and error.response is not None:
        try:
            response_body = error.response.json()
            if "detail" in response_body:
                error_msg += " " + str(response_body["detail"]).lower()
        except Exception:
            pass

    return error_msg


def is_token_limit_error(error_msg):
    return any(keyword in error_msg for keyword in TOKEN_LIMIT_KEYWORDS)


def parse_json_from_model_response(response):
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if not json_match:
            raise
        return json.loads(json_match.group())


def run_multiple_attempts(
    input_file,
    num_attempts=10,
    max_retries=2,
    trim_data=False,
    max_patient_json_chars=DEFAULT_MAX_PATIENT_JSON_CHARS,
):
    """Run model multiple times to generate candidate outputs."""
    prompts = load_json_file("./prompts/prompts.json")
    initial_prompt = prompts["initial"]

    original_patient_data = load_json_file(input_file)
    patient_data, patient_data_json = prepare_patient_data(
        original_patient_data,
        trim_data=trim_data,
        max_patient_json_chars=max_patient_json_chars,
    )

    full_prompt = f"{initial_prompt}\n\nPATIENT DATA:\n{patient_data_json}"

    attempts = []

    print(f"Generating {num_attempts} candidate outputs...")
    for i in range(num_attempts):
        print(f"  Attempt {i + 1}/{num_attempts}")

        try:
            model_response = call_model(full_prompt)
            parsed_json = parse_json_from_model_response(model_response)
            attempts.append(parsed_json)

        except Exception as e:
            error_msg = extract_error_message(e)

            if is_token_limit_error(error_msg):
                print(f"  ⚠ Token limit error detected: {error_msg}")
            elif isinstance(e, json.JSONDecodeError):
                print("Could not parse valid JSON from model response")
            else:
                print(f"  Error: {e}")

            continue

    if not attempts:
        raise ValueError("All attempts failed to produce valid JSON")

    print(f"Successfully generated {len(attempts)} outputs\n")

    return attempts, patient_data_json


def trim_patient_data(patient_data):
    """
    Trim patient data to keep only essential fields:
    patient_code, patient_name, species, breed, sex, prescriptions.
    """
    fields_to_keep = [
        "patient_code",
        "patient_name",
        "species",
        "breed",
        "sex",
        "prescriptions",
    ]

    trimmed_data = {}
    for field in fields_to_keep:
        if field in patient_data:
            trimmed_data[field] = patient_data[field]

    trimmed_data["_data_trimmed"] = True

    print(f"  Kept only essential fields: {', '.join(fields_to_keep)}")

    return trimmed_data


def calculate_antimicrobial_uncertainties(attempts):
    """
    Calculate uncertainty for each antimicrobial across all attempts.
    Returns a dictionary mapping antimicrobial names to their CV percentages.
    """
    antimicrobial_dosages = {}

    for attempt in attempts:
        if "antimicrobial_usage" not in attempt:
            continue

        for usage in attempt["antimicrobial_usage"]:
            drug_name = usage.get("antimicrobial_administered")
            dosage = usage.get("total_mg_administered")

            if drug_name and dosage is not None:
                try:
                    dosage_val = float(dosage)
                    if drug_name not in antimicrobial_dosages:
                        antimicrobial_dosages[drug_name] = []
                    antimicrobial_dosages[drug_name].append(dosage_val)
                except (ValueError, TypeError):
                    continue

    uncertainties = {}
    for drug_name, dosages in antimicrobial_dosages.items():
        if len(dosages) >= 2:
            dosages_array = np.array(dosages)
            mean_val = np.mean(dosages_array)
            std_val = np.std(dosages_array, ddof=1)
            cv_percent = (std_val / mean_val * 100) if mean_val != 0 else 0
            uncertainties[drug_name] = round(cv_percent, 2)
        else:
            uncertainties[drug_name] = None

    return uncertainties


def add_uncertainties_to_output(best_output, attempts):
    uncertainties = calculate_antimicrobial_uncertainties(attempts)

    if "antimicrobial_usage" in best_output:
        for usage in best_output["antimicrobial_usage"]:
            drug_name = usage.get("antimicrobial_administered")
            if drug_name and drug_name in uncertainties:
                usage["uncertainty_cv_percent"] = uncertainties[drug_name]
                print(f"  {drug_name}: CV = {uncertainties[drug_name]}%")

    return best_output


def select_best_with_model(attempts, patient_data_json):
    """Ask the model to evaluate and select the most internally consistent output."""
    prompts = load_json_file("./prompts/prompts.json")
    evaluation_template = prompts["evaluation"]

    evaluation_prompt = evaluation_template.format(
        num_attempts=len(attempts),
        patient_data=patient_data_json,
        attempts=json.dumps(attempts, indent=2),
        max_index=len(attempts) - 1,
    )

    print("Asking model to evaluate outputs for consistency...")
    response = call_model(evaluation_prompt, temperature=0.0)

    try:
        evaluation = parse_json_from_model_response(response)
        best_index = evaluation["best_output_index"]
        reasoning = evaluation["reasoning"]

        print(f"\n✓ Model selected output #{best_index + 1}")
        print(f"  Reasoning: {reasoning}\n")

        best_output = attempts[best_index]
        best_output = add_uncertainties_to_output(best_output, attempts)

        return best_output, evaluation

    except (json.JSONDecodeError, KeyError) as e:
        print("Could not parse valid JSON from model response")
        print(f"Error: {e}")

        best_output = attempts[0]
        best_output = add_uncertainties_to_output(best_output, attempts)

        return best_output, None


def process_single_patient(
    patient_code,
    num_attempts=10,
    trim_data=False,
    max_patient_json_chars=DEFAULT_MAX_PATIENT_JSON_CHARS,
):
    """Process a single patient file."""
    input_file = f"data/patients_with_ABX_2021/patient_{patient_code}.json"

    if not os.path.exists(input_file):
        print(f"⚠ Warning: Patient file not found: {input_file}")
        return None

    print(f"\n{'=' * 60}")
    print(f"Processing patient code: {patient_code}")
    print(f"File: {input_file}")
    print(f"{'=' * 60}\n")

    attempts, patient_data_json = run_multiple_attempts(
        input_file,
        num_attempts,
        trim_data=trim_data,
        max_patient_json_chars=max_patient_json_chars,
    )

    best_output, evaluation = select_best_with_model(attempts, patient_data_json)
    best_output["patient_code"] = patient_code

    return best_output


def main(
    num_patients,
    num_attempts=10,
    trim_data=False,
    max_patient_json_chars=DEFAULT_MAX_PATIENT_JSON_CHARS,
):
    """Main function to process N patients from mgadministered_gold_data.json."""
    master_file = "./data/mgadministered_gold_data.json"

    if not os.path.exists(master_file):
        print(f"Error: Master file '{master_file}' not found!")
        sys.exit(1)

    gold_data = load_json_file(master_file)
    patients_to_process = gold_data[:num_patients]

    print(f"\n{'#' * 60}")
    print(f"# Processing {len(patients_to_process)} patients from {master_file}")
    print(f"# Trim patient data: {trim_data}")
    print(f"# Max patient JSON chars: {max_patient_json_chars}")
    print(f"{'#' * 60}\n")

  
    all_results = []

    successful_count = 0
    failed_count = 0

    for i, patient_record in enumerate(patients_to_process, 1):
        patient_code = patient_record["code"]

        print(f"\n[{i}/{len(patients_to_process)}] Processing patient {patient_code}...")

        try:
            result = process_single_patient(
                patient_code,
                num_attempts,
                trim_data=trim_data,
                max_patient_json_chars=max_patient_json_chars,
            )

            if result:
                all_results.append(result)
                successful_count += 1
                print(f"✓ Successfully processed patient {patient_code}")
            else:
                failed_count += 1
                print(f"✗ Failed to process patient {patient_code}")

        except Exception as e:
            failed_count += 1
            print(f"✗ Error processing patient {patient_code}: {str(e)}")
            continue

    with open("./data/extracted_data.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("PROCESSING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total patients: {len(patients_to_process)}")
    print(f"Successful: {successful_count}")
    print(f"Failed: {failed_count}")
    print("\nResults saved to: ./data/extracted_data.json")
    print(f"{'=' * 60}\n")

    return all_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process patient JSON files and extract antimicrobial usage data."
    )
    parser.add_argument("num_patients", type=int)
    parser.add_argument("num_attempts", type=int, nargs="?", default=10)
    parser.add_argument(
        "--trim-data",
        action="store_true",
        help="Trim patient JSON to essential fields before sending it to the model.",
    )
    parser.add_argument(
        "--max-patient-json-chars",
        type=int,
        default=DEFAULT_MAX_PATIENT_JSON_CHARS,
        help="Maximum patient JSON characters to include before truncating.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = main(
        args.num_patients,
        args.num_attempts,
        trim_data=args.trim_data,
        max_patient_json_chars=args.max_patient_json_chars,
    )
