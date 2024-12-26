from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import os
import requests
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

gemini_api_key = os.environ.get("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found. Check .env file.")

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta2/models/gemini-pro:generateText"

def get_gemini_response(user_message):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {gemini_api_key}'
    }

    payload = {
        "prompt": {
            "text": user_message
        },
        "temperature": 0.7,
        "maxOutputTokens": 256
    }

    try:
        response = requests.post(GEMINI_API_URL, json=payload, headers=headers)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        if 'candidates' in data and data['candidates']:
            return data['candidates'][0].get('output', "No output generated.")
        elif 'error' in data:  # Handle Gemini API errors
            return f"Gemini API Error: {data['error']['message']}"
        else:
            return "Unexpected response format from Gemini API."

    except requests.exceptions.RequestException as e:
        return f"Request Error: {e}"
    except (KeyError, IndexError, TypeError) as e:
        return f"Parsing Error: {e}. Raw Response: {response.text if 'response' in locals() else 'No response received'}"

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message')
        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        bot_response = get_gemini_response(user_message)
        return jsonify({"reply": bot_response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)