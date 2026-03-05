import os
import requests

from dotenv import load_dotenv # type: ignore
load_dotenv()

API_URL = "https://ai-dashboard.vet.cornell.edu/api/models"
API_KEY = os.getenv("API_KEY")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

response = requests.get(API_URL, headers=headers)

if response.status_code == 200:
    models = response.json()
    print("Available models:")
    print(models)
else:
    print(f"Error: {response.status_code}")
    print(response.text)