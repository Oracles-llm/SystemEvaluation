import os
# from openai import OpenAI
# from dotenv import load_dotenv
import google.generativeai as genai

# load_dotenv()


class LLMJudge:
    def __init__(self):
        # self.api_key = os.getenv("OPENAI_API_KEY")
        # self.base_url = os.getenv("JUDGE_BASE_URL", "https://api.openai.com/v1")
        # self.model_name = os.getenv("JUDGE_MODEL_NAME", "gpt-4o")

        self.api_key = "AIzaSyDzjzvQy8WVcMu2_Etu3h0QtZFY4alaN28"
        self.model_name = os.getenv("JUDGE_MODEL_NAME", "gemini-1.5-flash")

        if not self.api_key or self.api_key == "AIzaSyDzjzvQy8WVcMu2_Etu3h0QtZFY4alaN28":
            print("WARNING: GEMINI API key not set in judges.py. Judge will fail.")

        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(self.model_name)

    def evaluate(self, prompt: str) -> str:
        """Raw call to the judge model"""
        try:
            response = self.client.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(temperature=0),
            )
            return response.text or ""
        except Exception as e:
            print(f"Judge Error: {e}")
            return "0.0"


# Singleton instance
judge_client = LLMJudge()
