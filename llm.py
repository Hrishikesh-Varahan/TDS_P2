import os
import openai
from dotenv import load_dotenv, find_dotenv

# Load .env for local dev
load_dotenv(find_dotenv())

AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
AIPROXY_BASE_URL = os.getenv("AIPROXY_BASE_URL")

if not AIPROXY_TOKEN or not AIPROXY_BASE_URL:
    raise RuntimeError("Missing LLM credentials in environment variables.")

client = openai.OpenAI(api_key=AIPROXY_TOKEN, base_url=AIPROXY_BASE_URL)

def get_llm_response(question: str, file_content: str = None):
    try:
        prompt = f"Answer concisely with final result only.\nQuestion: {question}\n"
        if file_content:
            prompt += f"\nContext:\n{file_content}"

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data analysis expert."},
                {"role": "user", "content": prompt}
            ]
        )
        answer = resp.choices[0].message.content.strip()
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}
