import os
import requests

from dotenv import load_dotenv # type: ignore
load_dotenv()

API_URL = "https://ai-dashboard.vet.cornell.edu/api/chat/completions"
API_KEY = os.getenv("API_KEY")


def call_model(prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    response = requests.post(API_URL, headers=headers, json=payload, timeout=200)

    if response.status_code != 200:
        print("STATUS:", response.status_code)
        print("BODY:", response.text) 
        response.raise_for_status()


    result = response.json()
    return result["choices"][0]["message"]["content"]
