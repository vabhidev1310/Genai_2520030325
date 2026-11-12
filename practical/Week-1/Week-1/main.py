import os

from dotenv import load_dotenv
from ollama import Client

load_dotenv()

model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
ollama_host = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434"
)

client = Client(host=ollama_host)

prompt = "What is Artificial Intelligence? Explain in simple words."

try:
    response = client.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    print("\nLLM Response:")
    print(response.message.content)

except Exception as error:
    print("\nSomething went wrong!")
    print("Error:", error)
    print("\nCheck whether Ollama is running and the model is installed.")