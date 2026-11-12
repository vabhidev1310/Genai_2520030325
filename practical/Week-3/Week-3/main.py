import os
from dotenv import load_dotenv

from llm_client import GeminiProvider, OllamaProvider, LLMClient
from mongo_logger import MongoLogger


def main():
    load_dotenv(override=True)

    logger = None

    try:
        print("\n=== LLM Monitoring System ===")
        print("1. Gemini")
        print("2. Ollama (Local Qwen)")

        choice = input("\nChoose your provider (1 or 2): ").strip()

        if choice == "1":
            api_key = os.getenv("GEMINI_API_KEY")

            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY is missing from your .env file."
                )

            provider = GeminiProvider(api_key)

        elif choice == "2":
            provider = OllamaProvider(model="qwen2.5:3b")

        else:
            print("Invalid choice. Please select 1 or 2.")
            return

        logger = MongoLogger()
        client = LLMClient(provider, logger)

        prompt = input("\nEnter your prompt: ").strip()

        if not prompt:
            print("Prompt cannot be empty.")
            return

        print("\nGenerating response...\n")
        result = client.generate(prompt)

        print("=== AI Response ===")
        print(result)

    except Exception as error:
        print(f"\nError: {error}")

    finally:
        if logger is not None:
            logger.close()


if __name__ == "__main__":
    main()
