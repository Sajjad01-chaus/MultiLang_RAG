# # app/llm_utils.py
# import os, requests
# from dotenv import load_dotenv

# load_dotenv()
# HF_API_KEY = os.getenv("HF_API_KEY", "")
# SMALL_LLM_MODEL = os.getenv("SMALL_LLM_MODEL", "microsoft/DialoGPT-medium")

# # In app/llm_utils.py

# def call_llm(prompt: str) -> str:
#     """Call HuggingFace Inference API for LLM responses with robust error handling."""
#     if not HF_API_KEY:
#         return "[ERROR] HuggingFace API key not configured. Please set HF_API_KEY environment variable."

#     API_URL = f"https://api-inference.huggingface.co/models/{SMALL_LLM_MODEL}"
#     headers = {"Authorization": f"Bearer {HF_API_KEY}"}
#     payload = {
#         "inputs": prompt,
#         "parameters": {
#             "max_new_tokens": 250,
#             "return_full_text": False,
#             "temperature": 0.7
#         }
#     }

#     try:
#         r = requests.post(API_URL, headers=headers, json=payload, timeout=45) # Increased timeout slightly

#         # Check if the response is successful and has content
#         if r.status_code == 200 and r.content:
#             # ✅ FIX: Safely try to parse the JSON response.
#             try:
#                 data = r.json()
#                 if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
#                     return data[0]["generated_text"].strip()
#                 else:
#                     return f"[HF API Error] Unexpected response format: {data}"
#             # ✅ FIX: This is the key part. Catch the error if the response is not valid JSON.
#             except requests.exceptions.JSONDecodeError:
#                 return "[HF API Error] Received an invalid response (not JSON). The model might be loading or the API is overloaded. Please try again in a moment."
#         else:
#             # Handle non-200 status codes
#             error_details = r.text
#             try:
#                 # Try to get a cleaner error message if the response is JSON
#                 error_details = r.json().get("error", r.text)
#             except requests.exceptions.JSONDecodeError:
#                 pass # Stick with the raw text if it's not JSON
#             return f"[HF API Error] {r.status_code}: {error_details}"

#     except requests.exceptions.Timeout:
#         return "[ERROR] HuggingFace API request timed out. The model may be taking too long to load."
#     except requests.exceptions.RequestException as e:
#         return f"[ERROR] HuggingFace API request failed: {e}"

# app/llm_utils.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dotenv import load_dotenv

load_dotenv()

# Get the model ID from your .env file
MODEL_ID = os.getenv("SMALL_LLM_MODEL", "google/mt5-base")

# --- LOCAL MODEL INITIALIZATION ---
# This code runs only once when your application starts.
# It downloads the model and prepares it for use.
print("---")
print(f"Attempting to load local LLM: {MODEL_ID}")
LLM_TOKENIZER = None
LLM_MODEL = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    # Load the tokenizer and the model itself
    LLM_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
    LLM_MODEL = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(DEVICE)
    print(f"Successfully loaded local LLM to device: {DEVICE}")
except Exception as e:
    print(f"FATAL ERROR: Could not load the local LLM. Please check model name and internet connection.")
    print(e)
print("---")
# --- END OF INITIALIZATION ---


def call_llm(prompt: str) -> str:
    """
    Generates a response using the locally-hosted Hugging Face model.
    This bypasses the unreliable HF Inference API.
    """
    if not LLM_MODEL or not LLM_TOKENIZER:
        return "[ERROR] The local LLM could not be loaded, so no answer can be generated."

    try:
        # Prepare the prompt for the model
        inputs = LLM_TOKENIZER(prompt, return_tensors="pt").to(DEVICE)

        # Generate a response from the model
        outputs = LLM_MODEL.generate(
            **inputs,
            max_new_tokens=250,  # Controls the max length of the answer
            temperature=0.7
        )

        # Decode the response back into text
        response_text = LLM_TOKENIZER.batch_decode(outputs, skip_special_tokens=True)[0]
        return response_text

    except Exception as e:
        print(f"Error during local LLM generation: {e}")
        return "[ERROR] An unexpected error occurred while generating the answer with the local model."