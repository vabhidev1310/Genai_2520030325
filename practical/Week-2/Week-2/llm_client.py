from abc import ABC, abstractmethod

import requests
from google import genai


# 1. Common interface
class LLMProvider(ABC):

    @abstractmethod
    def generate(self, prompt):
        pass


# 2. Gemini cloud provider
class GeminiProvider(LLMProvider):

    def __init__(self, api_key, model="gemini-3.8-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(self, prompt):

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        if response.text is None:
            raise RuntimeError("Gemini returned no text response.")

        return response.text


# 3. Ollama local provider
class OllamaProvider(LLMProvider):

    def __init__(
        self,
        model="qwen2.5:3b",
        url="http://localhost:11434/api/generate"
    ):
        self.model = model
        self.url = url

    def generate(self, prompt):

        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()

        return response.json()["response"]


# 4. Common LLM client
class LLMClient:

    def __init__(self, provider):
        self.provider = provider

    def generate(self, prompt):
        return self.provider.generate(prompt)
