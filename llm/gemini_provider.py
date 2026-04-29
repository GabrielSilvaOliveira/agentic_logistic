import os
import google.generativeai as genai
import time
from dotenv import load_dotenv
from llm.base_llm import BaseLLM

load_dotenv()


class GeminiProvider(BaseLLM):

    def __init__(self, model_name: str = "gemini-2.5-flash"):

        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel(model_name)

    def generate(self, prompt: str, system_prompt: str = None) -> str:

        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt

        response = self.model.generate_content(full_prompt)

        return response.text
    
    def safe_generate(self, prompt):

        for _ in range(3):

            try:
                response = self.model.generate_content(prompt)
                return response.text

            except Exception:
                time.sleep(2)

        raise RuntimeError("LLM request failed")