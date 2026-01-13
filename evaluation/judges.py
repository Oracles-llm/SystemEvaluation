import os
import re
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
        self.model_name = os.getenv("JUDGE_MODEL_NAME", "gemini-1.5-pro")
        self.fallback_models = [self.model_name]
        if self.model_name != "gemini-1.5-flash":
            self.fallback_models.append("gemini-1.5-flash")
        self.last_error = None
        self.last_error_code = None
        self.last_model_used = None
        self.last_model_fallback = False

        if not self.api_key or self.api_key == "AIzaSyDzjzvQy8WVcMu2_Etu3h0QtZFY4alaN28":
            print("WARNING: GEMINI API key not set in judges.py. Judge will fail.")

        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(self.model_name)

    def _extract_status_code(self, error: Exception) -> int | None:
        for attr in ("status_code", "code"):
            value = getattr(error, attr, None)
            if isinstance(value, int):
                return value
            if hasattr(value, "value") and isinstance(value.value, int):
                return value.value
        match = re.search(r"\b(\d{3})\b", str(error))
        if match:
            return int(match.group(1))
        return None

    def evaluate(self, prompt: str) -> str:
        """Raw call to the judge model"""
        self.last_error = None
        self.last_error_code = None
        self.last_model_used = None
        self.last_model_fallback = False
        for idx, model_name in enumerate(self.fallback_models):
            self.model_name = model_name
            self.client = genai.GenerativeModel(self.model_name)
            self.last_model_fallback = idx > 0
            try:
                response = self.client.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(temperature=0),
                )
                self.last_model_used = self.model_name
                return response.text or ""
            except Exception as e:
                status_code = self._extract_status_code(e)
                if status_code == 404 and idx < len(self.fallback_models) - 1:
                    continue
                self.last_error = str(e)
                self.last_error_code = status_code
                self.last_model_used = self.model_name
                print(f"Judge Error: {e}")
                return ""


# Singleton instance
judge_client = LLMJudge()
