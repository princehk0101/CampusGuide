import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.messages import AIMessage

from backend.documents import (
    MAX_PDF_BYTES,
    answer_pdf_question,
    get_pdf_content_and_retriever,
    get_pdf_pages,
    summarize_pdf,
)
from backend.university_sources import UNIVERSITY_DATA_DIR
from backend.main import app


class FakeRetriever:
    def invoke(self, question: str) -> list[Document]:
        return [
            Document(
                page_content="The application deadline is 30 June.",
                metadata={"page": 1},
            )
        ]


class FakeChatModel:
    def invoke(self, messages: list) -> AIMessage:
        return AIMessage(
            content="The deadline is 30 June. [Uploaded PDF, p. 2]"
        )


class DocumentQaTests(unittest.TestCase):
    def test_rejects_non_pdf_or_invalid_pdf_header(self) -> None:
        with self.assertRaisesRegex(ValueError, "upload a PDF"):
            get_pdf_content_and_retriever("notes.txt", b"not a pdf")
        with self.assertRaisesRegex(ValueError, "valid PDF header"):
            get_pdf_content_and_retriever("notes.pdf", b"not a pdf")

    def test_rejects_upload_over_size_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "15 MB or smaller"):
            get_pdf_content_and_retriever(
                "large.pdf",
                b"%PDF-" + b"x" * MAX_PDF_BYTES,
            )

    def test_extracts_text_from_official_bu_bulletin(self) -> None:
        content = (
            UNIVERSITY_DATA_DIR / "admission_bulletin_2026_27.pdf"
        ).read_bytes()

        pages = get_pdf_pages("admission_bulletin_2026_27.pdf", content)

        self.assertGreaterEqual(len(pages), 1)
        self.assertTrue(
            any(
                "BUNDELKHAND UNIVERSITY" in page.page_content
                for page in pages
            )
        )

    def test_indexes_uploaded_pdf_and_returns_passages(self) -> None:
        content = (
            UNIVERSITY_DATA_DIR / "admission_bulletin_2026_27.pdf"
        ).read_bytes()
        with patch(
            "backend.documents.get_embeddings",
            return_value=FakeEmbeddings(size=8),
        ):
            _, retriever = get_pdf_content_and_retriever(
                "admission_bulletin_2026_27.pdf",
                content,
            )

        matches = retriever.invoke("2026-27 academic bulletin and fee")
        self.assertTrue(matches)
        self.assertTrue(
            any("page" in document.metadata for document in matches)
        )

    def test_question_answer_includes_retrieved_page_citation(self) -> None:
        answer, citations = answer_pdf_question(
            FakeChatModel(),
            FakeRetriever(),
            "When is the application deadline?",
        )

        self.assertIn("30 June", answer)
        self.assertEqual(citations, ["[Uploaded PDF, p. 2]"])

    def test_summary_preserves_source_citation(self) -> None:
        summary, citations = summarize_pdf(
            FakeChatModel(),
            (
                Document(
                    page_content="The application deadline is 30 June.",
                    metadata={"page": 1},
                ),
            ),
        )

        self.assertIn("30 June", summary)
        self.assertEqual(citations, ["[Uploaded PDF, p. 2]"])

    def test_summary_groups_pages_and_cites_each_page(self) -> None:
        summary, citations = summarize_pdf(
            FakeChatModel(),
            (
                Document(
                    page_content="Page one information.",
                    metadata={"page": 0},
                ),
                Document(
                    page_content="Page two information.",
                    metadata={"page": 1},
                ),
            ),
        )

        self.assertIn("[Uploaded PDF, p. 1]", summary)
        self.assertIn("[Uploaded PDF, p. 2]", summary)
        self.assertEqual(
            citations,
            ["[Uploaded PDF, p. 1]", "[Uploaded PDF, p. 2]"],
        )

    def test_question_answer_adds_citation_if_model_omits_it(self) -> None:
        class UncitedChatModel:
            def invoke(self, messages: list) -> AIMessage:
                return AIMessage(content="The deadline is 30 June.")

        answer, citations = answer_pdf_question(
            UncitedChatModel(),
            FakeRetriever(),
            "When is the application deadline?",
        )

        self.assertIn("Sources: [Uploaded PDF, p. 2]", answer)
        self.assertEqual(citations, ["[Uploaded PDF, p. 2]"])

    def test_upload_api_requires_key_and_returns_question_answer(self) -> None:
        os.environ["CAMPUSGUIDE_API_KEY"] = "document-test-key"
        with (
            patch(
                "backend.main.get_document_llm",
                return_value=FakeChatModel(),
            ),
            patch(
                "backend.main.get_pdf_content_and_retriever",
                return_value=((Document(
                    page_content="Deadline text",
                    metadata={"page": 1},
                ),), FakeRetriever()),
            ) as build_index,
            patch(
                "backend.main.answer_pdf_question",
                return_value=(
                    "Deadline is 30 June. [Uploaded PDF, p. 2]",
                    ["[Uploaded PDF, p. 2]"],
                ),
            ),
        ):
            with TestClient(app) as client:
                payload = {
                    "file": ("handbook.pdf", b"%PDF-test", "application/pdf"),
                }
                denied = client.post(
                    "/api/documents/analyze",
                    files=payload,
                    data={"mode": "question", "question": "Deadline?"},
                )
                self.assertEqual(denied.status_code, 401)

                response = client.post(
                    "/api/documents/analyze",
                    headers={"X-CampusGuide-Key": "document-test-key"},
                    files=payload,
                    data={"mode": "question", "question": "Deadline?"},
                )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["mode"], "question")
        self.assertEqual(
            response.json()["citations"],
            ["[Uploaded PDF, p. 2]"],
        )
        self.assertEqual(build_index.call_args.args[1], b"%PDF-test")

    def test_upload_api_returns_pdf_summary(self) -> None:
        os.environ["CAMPUSGUIDE_API_KEY"] = "document-test-key"
        pages = (
            Document(
                page_content="The application deadline is 30 June.",
                metadata={"page": 1},
            ),
        )
        with (
            patch(
                "backend.main.get_document_llm",
                return_value=FakeChatModel(),
            ),
            patch("backend.main.get_pdf_pages", return_value=pages),
            patch(
                "backend.main.summarize_pdf",
                return_value=(
                    "Deadline: 30 June. [Uploaded PDF, p. 2]",
                    ["[Uploaded PDF, p. 2]"],
                ),
            ),
        ):
            with TestClient(app) as client:
                response = client.post(
                    "/api/documents/analyze",
                    headers={"X-CampusGuide-Key": "document-test-key"},
                    files={
                        "file": (
                            "handbook.pdf",
                            b"%PDF-test",
                            "application/pdf",
                        ),
                    },
                    data={"mode": "summary"},
                )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["mode"], "summary")
        self.assertIn("30 June", response.json()["answer"])
        self.assertEqual(
            response.json()["citations"],
            ["[Uploaded PDF, p. 2]"],
        )

    def test_upload_api_rejects_empty_question(self) -> None:
        os.environ["CAMPUSGUIDE_API_KEY"] = "document-test-key"
        with patch("backend.main.get_document_llm") as get_llm:
            with TestClient(app) as client:
                response = client.post(
                    "/api/documents/analyze",
                    headers={"X-CampusGuide-Key": "document-test-key"},
                    files={
                        "file": (
                            "handbook.pdf",
                            b"%PDF-test",
                            "application/pdf",
                        ),
                    },
                    data={"mode": "question", "question": "  "},
                )

        self.assertEqual(response.status_code, 422)
        get_llm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
