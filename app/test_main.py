import os
import unittest
import importlib
import io

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from fastapi import UploadFile
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


def _make_upload_file(content: bytes, filename: str = "doc.txt") -> UploadFile:
    return UploadFile(filename=filename, file=io.BytesIO(content))


def _fake_llm_response(answer: str = "mock answer") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=answer),
            )
        ]
    )


class _FakeVectorStore:
    def similarity_search(self, _question: str, k: int = 3):
        return [SimpleNamespace(page_content="context", metadata={"source": "doc.txt"}) for _ in range(k)]


class TestVectorStoreCache(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        import app.config as config
        import app.rag as rag

        importlib.reload(config)
        importlib.reload(rag)
        self.rag = rag
        self.settings = config.settings

        with self.rag._VECTORSTORE_CACHE_LOCK:
            self.rag._VECTORSTORE_CACHE.clear()

    def tearDown(self) -> None:
        with self.rag._VECTORSTORE_CACHE_LOCK:
            self.rag._VECTORSTORE_CACHE.clear()

    async def test_cache_hit_reuses_vectorstore(self):
        fake_vectorstore = _FakeVectorStore()

        with (
            mock.patch("app.rag.HuggingFaceEndpointEmbeddings", return_value=object()),
            mock.patch("app.rag.InMemoryVectorStore.from_documents", return_value=fake_vectorstore) as from_documents,
            mock.patch("app.rag.InferenceClient") as inference_client,
        ):
            inference_client.return_value.chat_completion.return_value = _fake_llm_response()

            response_1 = await self.rag.run_rag_query(
                file=_make_upload_file(b"same document content"),
                question="What is this?",
                top_k=2,
                stream=False,
                hf_token="hf_abcdefghijklmnopqrstuvwxyz1234",
                settings=self.settings,
            )
            self.assertEqual(response_1.status_code, 200)

            response_2 = await self.rag.run_rag_query(
                file=_make_upload_file(b"same document content"),
                question="What is this?",
                top_k=2,
                stream=False,
                hf_token="hf_abcdefghijklmnopqrstuvwxyz1234",
                settings=self.settings,
            )
            self.assertEqual(response_2.status_code, 200)

            self.assertEqual(from_documents.call_count, 1)

    async def test_cache_miss_rebuilds_vectorstore_for_different_document(self):
        fake_vectorstore = _FakeVectorStore()

        with (
            mock.patch("app.rag.HuggingFaceEndpointEmbeddings", return_value=object()),
            mock.patch("app.rag.InMemoryVectorStore.from_documents", return_value=fake_vectorstore) as from_documents,
            mock.patch("app.rag.InferenceClient") as inference_client,
        ):
            inference_client.return_value.chat_completion.return_value = _fake_llm_response()

            response_1 = await self.rag.run_rag_query(
                file=_make_upload_file(b"document content one"),
                question="What is this?",
                top_k=2,
                stream=False,
                hf_token="hf_abcdefghijklmnopqrstuvwxyz1234",
                settings=self.settings,
            )
            self.assertEqual(response_1.status_code, 200)

            response_2 = await self.rag.run_rag_query(
                file=_make_upload_file(b"document content two"),
                question="What is this?",
                top_k=2,
                stream=False,
                hf_token="hf_abcdefghijklmnopqrstuvwxyz1234",
                settings=self.settings,
            )
            self.assertEqual(response_2.status_code, 200)

            self.assertEqual(from_documents.call_count, 2)
