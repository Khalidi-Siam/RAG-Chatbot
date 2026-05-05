import os
from google import genai
from logger import logging


class GeminiLLM:
    def __init__(self, api_key: str = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError("Gemini API Key not found. Set GOOGLE_API_KEY env var.")

        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        logging.info(f"GeminiLLM initialized with model '{self.model_name}'")

    def generate(self, prompt: str) -> str:
        """
        Generate response from Gemini model.
        """
        logging.info(f"Sending prompt to '{self.model_name}' (prompt length: {len(prompt)} chars)")
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )

        result = response.text.strip() if response.text else ""
        logging.info(f"Response received ({len(result)} chars)")
        return result