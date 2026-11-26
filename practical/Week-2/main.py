import os

from dotenv import load_dotenv

from llm_client import (
    GeminiProvider,
    OllamaProvider,
    LLMClient
)


# Load environment variables
load_dotenv()

print("\n===== PROVIDER-AGNOSTIC LLM CLIENT =====")
print("1. Gemini (Cloud)")
print("2. Ollama (Local)")

provider_name = input(
    "\nChoose provider (gemini/ollama): "
).strip().lower()


# Select provider
if provider_name == "gemini":

    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

    if not api_key:
        raise ValueError(
            "Gemini API key not found in .env"
        )

    provider = GeminiProvider(
        api_key=api_key,
        model=model
    )

elif provider_name == "ollama":

    model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

    url = os.getenv(
        "OLLAMA_URL",
        "http://localhost:11434/api/generate"
    )

    provider = OllamaProvider(
        model=model,
        url=url
    )

else:
    raise ValueError("Invalid provider selected")


# Initialize common client
client = LLMClient(provider)

# Get prompt
prompt = input("\nEnter your question: ")

# Generate response
answer = client.generate(prompt)

# Display response
print("\n===== LLM RESPONSE =====")
print(answer)
