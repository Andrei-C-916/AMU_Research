import json
import os
from open_webui_client import call_model

def main(input_file):
    """Main function to process patient data"""
    
    # Load the prompt from prompts.json
    with open('./prompts.json', 'r') as f:
        prompts = json.load(f)
        initial_prompt = prompts['initial']
    
    # Load the patient data from input file
    with open(input_file, 'r') as f:
        patient_data = json.load(f)
    
    # Combine prompt with patient data
    full_prompt = f"{initial_prompt}\n\nPATIENT DATA:\n{json.dumps(patient_data, indent=2)}"
    
    # Call the model
    print("Calling model...")
    model_response = call_model(full_prompt)
    
    # Parse the response to ensure it's valid JSON
    try:
        # Try to extract JSON from the response (in case there's extra text)
        parsed_json = json.loads(model_response)
        print("Successfully parsed JSON response")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"Model response: {model_response}")
        # Try to find JSON in the response if it's wrapped in text
        import re
        json_match = re.search(r'\{.*\}', model_response, re.DOTALL)
        if json_match:
            try:
                parsed_json = json.loads(json_match.group())
                print("Successfully extracted JSON from response")
            except json.JSONDecodeError:
                raise ValueError("Could not parse valid JSON from model response")
        else:
            raise ValueError("No JSON found in model response")
    
    # Load existing results if file exists
    if os.path.exists('out.json'):
        with open('out.json', 'r') as f:
            all_results = json.load(f)
    else:
        all_results = []

# Append new result
    all_results.append(parsed_json)

# Write back
    with open('out.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("Results saved to out.json")
    return parsed_json

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_file.json>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    result = main(input_file)
    print(json.dumps(result, indent=2))