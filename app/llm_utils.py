import os, time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL_ID = os.getenv("OPENROUTER_MODEL_ID", "google/gemma-3n-e2b-it:free")
FALLBACK_MODEL_ID = "mistralai/mistral-7b-instruct:free"
OPENROUTER_REFERRER = "https://github.com/Sajjad01-chaus/MultiLang_RAG"

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

def call_llm(prompt: str) -> str:
    if not OPENROUTER_API_KEY:
        return "[ERROR] Missing OpenRouter API key."

    def make_request(model_id):
        return client.chat.completions.create(
            extra_headers={"HTTP-Referer": OPENROUTER_REFERRER},
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1536,
            temperature=0.6,
        )

    try:
        completion = make_request(OPENROUTER_MODEL_ID)
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"[LLM] Error: {e}")
        if "429" in str(e):
            time.sleep(60)
            try:
                completion = make_request(FALLBACK_MODEL_ID)
                print(f"[LLM] Fallback model used ({FALLBACK_MODEL_ID}) ✅")
                return completion.choices[0].message.content.strip()
            except Exception as e2:
                return f"[API ERROR] Fallback failed: {e2}"
        return f"[API ERROR] {e}"
