import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class LLMJudge:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("JUDGE_BASE_URL", "https://api.openai.com/v1")
        self.model_name = os.getenv("JUDGE_MODEL_NAME", "gpt-4o")
        
        if not self.api_key:
            print("⚠️ WARNING: OPENAI_API_KEY not found in .env. Judge will fail.")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def evaluate(self, prompt: str) -> str:
        """Raw call to the judge model"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Judge Error: {e}")
            return "0.0"

# Singleton instance
judge_client = LLMJudge()