
import os
from dotenv import load_dotenv

from usage_logger import UsageLogger
from llm_client import (
    LLMClient,
    GeminiProvider,
    OllamaProvider
)

load_dotenv(override=True)


def main():
    logger = None

    try:
        logger = UsageLogger()

        print("\nChoose your LLM provider:")
        print("1. Gemini")
        print("2. Ollama")

        choice = input("Enter your choice (1 or 2): ")

        if choice == "1":
            provider = GeminiProvider()
        elif choice == "2":
            provider = OllamaProvider()
        else:
            print("Invalid choice!")
            return

        client = LLMClient(provider, logger)

        prompt = input("\nEnter your prompt: ")

        print("\nGenerating response...\n")

        result = client.generate(prompt)

        print("Response:")
        print(result.text)

        print("\nToken Usage:")
        print("Input tokens:", result.input_tokens)
        print("Output tokens:", result.output_tokens)
        print(
            "Total tokens:",
            result.input_tokens + result.output_tokens
        )

    except Exception as error:
        print("Error:", error)

    finally:
        if logger:
            logger.close()


if __name__ == "__main__":
    main()
