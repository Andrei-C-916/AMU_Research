import json
import os
from open_webui_client import call_model
import re

def run_multiple_attempts(input_file, num_attempts=10):
    """Run model multiple times to generate candidate outputs"""
    
    # Load prompt and patient data
    with open('./prompts.json', 'r') as f:
        prompts = json.load(f)
        initial_prompt = prompts['initial']
    
    with open(input_file, 'r') as f:
        patient_data = json.load(f)
    
    full_prompt = f"{initial_prompt}\n\nPATIENT DATA:\n{json.dumps(patient_data, indent=2)}"
    
    # Collect all successful attempts
    attempts = []
    
    print(f"Generating {num_attempts} candidate outputs...")
    for i in range(num_attempts):
        print(f"  Attempt {i+1}/{num_attempts}")
        
        try:
            model_response = call_model(full_prompt)
            parsed_json = json.loads(model_response)
            attempts.append(parsed_json)
            
        except json.JSONDecodeError as e:
            json_match = re.search(r'\{.*\}', model_response, re.DOTALL)
            if json_match:
                try:
                    parsed_json = json.loads(json_match.group())
                    attempts.append(parsed_json)
                except:
                    print("Could not parse valid JSON from model response")
            else:
                print("No JSON found in model response")
            continue
    
    if not attempts:
        raise ValueError("All attempts failed to produce valid JSON")
    
    print(f"Successfully generated {len(attempts)} outputs\n")
    return attempts, patient_data

def select_best_with_model(attempts, patient_data):
    """Ask the model to evaluate and select the most internally consistent output"""
    with open('./prompts.json', 'r') as f:
        prompts = json.load(f)
        evaluation_template = prompts['evaluation']

    evaluation_prompt = evaluation_template.format(
        num_attempts=len(attempts),
        patient_data=json.dumps(patient_data, indent=2),
        attempts=json.dumps(attempts, indent=2),
        max_index=len(attempts) - 1
    )    

    print("Asking model to evaluate outputs for consistency...")
    response = call_model(evaluation_prompt, temperature=0.0)
    
    try:
        evaluation = json.loads(response)
        best_index = evaluation['best_output_index']
        reasoning = evaluation['reasoning']
        
        print(f"\n✓ Model selected output #{best_index + 1}")
        print(f"  Reasoning: {reasoning}\n")
        
        return attempts[best_index], evaluation
    
    except (json.JSONDecodeError, KeyError) as e:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                evaluation = json.loads(json_match.group())
                best_index = evaluation['best_output_index']
                reasoning = evaluation['reasoning']

                print(f"\n✓ Model selected output #{best_index + 1}")
                print(f"  Reasoning: {reasoning}\n")

                return attempts[best_index], evaluation
            except (json.JSONDecodeError, KeyError) as e:
                print("Could not parse valid JSON from model response")
                print(f"Error: {e}")
                return attempts[0], None
        else:
            print("No JSON found in model response")
            print(f"Error: {e}")
            return attempts[0], None

def main(input_file, num_attempts=10):
    """Main function to process patient data with self-consistency"""
    
    # Generate multiple candidate outputs
    attempts, patient_data = run_multiple_attempts(input_file, num_attempts)
    
    # Have model select the best one
    best_output, evaluation = select_best_with_model(attempts, patient_data)
    
    # Save all attempts for analysis (optional)
    # with open('all_attempts.json', 'w') as f:
    #     json.dump({
    #         'attempts': attempts,
    #         'evaluation': evaluation,
    #         'selected_output': best_output
    #     }, f, indent=2)
    
    # Load existing results if file exists
    if os.path.exists('extracted_data.json'):
        with open('extracted_data.json', 'r') as f:
            all_results = json.load(f)
    else:
        all_results = []
    
    # Append new result
    all_results.append(best_output)
    
    # Write back
    with open('extracted_data.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("Best result saved to extracted_data.json")
    print("All attempts saved to all_attempts.json")
    
    return best_output

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_file.json> [num_attempts]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    num_attempts = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    result = main(input_file, num_attempts)
    print(json.dumps(result, indent=2))