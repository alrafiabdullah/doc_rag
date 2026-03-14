import os
import unittest
import importlib

from pathlib import Path
from unittest import mock

from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


ROOT_DIR = Path(__file__).resolve().parents[1]
if load_dotenv is not None:
    load_dotenv(ROOT_DIR / ".env", override=False)

VALID_TOKEN = os.getenv("HUGGINGFACE_TEST_API_KEY")


class TestFastAPIApp(unittest.TestCase):
    def setUp(self) -> None:
        self.main = self._reload_modules()

        async def fake_run_rag_query(**kwargs):
            return JSONResponse(
                {
                    "answer": "mocked answer",
                    "meta": {
                        "retrieved_chunks": 1,
                        "question_chars": len(kwargs.get("question", "")),
                        "accepted_file_types": [".txt", ".pdf"],
                        "file_persisted": False,
                    },
                    "sources": [
                        {
                            "rank": 1,
                            "source": "test.txt",
                            "preview": "preview",
                        }
                    ],
                }
            )

        self._patcher = mock.patch.object(self.main, "run_rag_query", new=fake_run_rag_query)
        self._patcher.start()
        self.client = TestClient(self.main.app)

    def tearDown(self) -> None:
        self._patcher.stop()

    @staticmethod
    def _reload_modules():
        import app.config as config
        import app.main as main

        importlib.reload(config)
        importlib.reload(main)
        return main

    def test_query_requires_token(self):
        with mock.patch.dict(
            os.environ,
            {
                "HUGGINGFACE_API_KEY": "",
                "HF_TOKEN": "",
                "HUGGINGFACEHUB_API_TOKEN": "",
            },
            clear=False,
        ):
            response = self.client.post(
                "/rag/query",
                files={"file": ("test.txt", b"hello", "text/plain")},
                data={"question": "What is this?"},
            )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Missing Hugging Face token")

    def test_query_success_with_authorization_header(self):
        response = self.client.post(
            "/rag/query",
            files={"file": ("test.txt", b"hello", "text/plain")},
            data={"question": "What is this?", "top_k": "2", "stream": "false"},
            headers={"Authorization": f"Bearer {VALID_TOKEN}"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["answer"], "mocked answer")

    def test_rate_limit_returns_429(self):
        self.main.rate_limiter = self.main.InMemoryRateLimiter(window_seconds=300, max_requests=2)

        for _ in range(2):
            ok = self.client.post(
                "/rag/query",
                files={"file": ("test.txt", b"hello", "text/plain")},
                data={"question": "What is this?", "stream": "false"},
                headers={"Authorization": f"Bearer {VALID_TOKEN}"},
            )
            self.assertEqual(ok.status_code, 200)

        blocked = self.client.post(
            "/rag/query",
            files={"file": ("test.txt", b"hello", "text/plain")},
            data={"question": "What is this?", "stream": "false"},
            headers={"Authorization": f"Bearer {VALID_TOKEN}"},
        )

        self.assertEqual(blocked.status_code, 429)
        self.assertIn("Rate limit exceeded", blocked.json()["detail"])


if __name__ == "__main__":
    unittest.main()
