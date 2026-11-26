
import os
import time
import requests
from abc import ABC, abstractmethod
from dataclasses import dataclass
from google import genai
from google.genai import types


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt):
        pass


class GeminiProvider(LLMProvider):
    def __init__(self):
        self.client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY"),
            http_options=types.HttpOptions(timeout=60000)
        )
        self.model = "gemini-3.8-flash"

    def generate(self, prompt):
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        usage = response.usage_metadata

        return LLMResult(
            text=response.text or "",
            input_tokens=usage.prompt_token_count or 0,
            output_tokens=usage.candidates_token_count or 0
        )


class OllamaProvider(LLMProvider):
    def __init__(self):
        self.model = "llama3.2"
        self.url = "http://localhost:11434/api/generate"

    def generate(self, prompt):
        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            },
            timeout=(5, 60)
        )

        response.raise_for_status()
        data = response.json()

        return LLMResult(
            text=data.get("response", ""),
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0)
        )


class LLMClient:
    def __init__(self, provider, logger):
        self.provider = provider
        self.logger = logger

        self.input_rate = float(
            os.getenv("GEMINI_INPUT_COST_PER_MILLION", "0.30")
        )
        self.output_rate = float(
            os.getenv("GEMINI_OUTPUT_COST_PER_MILLION", "2.50")
        )

    def generate(self, prompt):
        for attempt in range(1, 4):
            start = time.perf_counter()

            try:
                result = self.provider.generate(prompt)

                latency = time.perf_counter() - start

                if isinstance(self.provider, GeminiProvider):
                    cost = (
                        result.input_tokens * self.input_rate
                        + result.output_tokens * self.output_rate
                    ) / 1_000_000
                else:
                    cost = 0.0

                self.logger.log_call({
                    "provider": type(self.provider).__name__,
                    "model": getattr(self.provider, "model", "unknown"),
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "total_tokens": (
                        result.input_tokens + result.output_tokens
                    ),
                    "latency_seconds": round(latency, 3),
                    "estimated_cost": round(cost, 8),
                    "status": "success",
                    "attempt": attempt
                })

                return result

            except Exception as error:
                latency = time.perf_counter() - start

                self.logger.log_call({
                    "provider": type(self.provider).__name__,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "latency_seconds": round(latency, 3),
                    "estimated_cost": 0.0,
                    "status": "failed",
                    "error": str(error),
                    "attempt": attempt
                })

                error_text = str(error).lower()
                temporary = any(
                    word in error_text
                    for word in [
                        "timeout", "429", "500", "502",
                        "503", "504", "connection"
                    ]
                )

                if not temporary or attempt == 3:
                    raise

                time.sleep(2 ** (attempt - 1))
