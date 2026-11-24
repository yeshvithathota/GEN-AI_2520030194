
import unittest
from unittest.mock import Mock

from llm_client import LLMClient, LLMResult


class FakeProvider:
    def generate(self, prompt):
        return LLMResult(
            text="Hello! This is a test response.",
            input_tokens=5,
            output_tokens=8
        )


class TestLLMClient(unittest.TestCase):

    def test_successful_generation(self):
        fake_logger = Mock()

        client = LLMClient(
            provider=FakeProvider(),
            logger=fake_logger
        )

        result = client.generate("Hello")

        self.assertEqual(
            result.text,
            "Hello! This is a test response."
        )

        fake_logger.log_call.assert_called_once()

        logged_record = fake_logger.log_call.call_args[0][0]

        self.assertEqual(logged_record["total_tokens"], 13)
        self.assertEqual(logged_record["status"], "success")


if __name__ == "__main__":
    unittest.main()
