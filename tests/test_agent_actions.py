from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import ANY, MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, ToolMessage

from backend.agent import (
    BTECH_CSE_SYLLABUS_URL,
    SYLLABUS_INDEX_URL,
    _append_syllabus_index_link,
    _append_btech_cse_syllabus_pdf,
    get_cited_document_source_links,
    _is_syllabus_request,
    _university_source_fingerprint,
    _search_budget_exhausted,
    build_retriever,
    summarize_agent_actions,
)


class AgentActionSummaryTests(unittest.TestCase):
    def test_syllabus_questions_always_get_official_index_link(self) -> None:
        answer = _append_syllabus_index_link(
            "I could not confirm the semester-specific syllabus.",
            "MCA syllabus",
        )

        self.assertIn(SYLLABUS_INDEX_URL, answer)
        self.assertIn("[Open the programme-wise syllabus index]", answer)
        self.assertIn(
            "check the programme and session",
            answer.casefold(),
        )

    def test_syllabus_link_is_not_duplicated(self) -> None:
        answer = (
            f"See the official syllabus index: {SYLLABUS_INDEX_URL}"
        )

        self.assertEqual(
            _append_syllabus_index_link(answer, "MCA syllabus"),
            answer,
        )

    def test_btech_cse_syllabus_gets_direct_pdf_and_session_caveat(self) -> None:
        answer = _append_btech_cse_syllabus_pdf(
            "The assistant claims the subjects are on page 14 and says this "
            "is confirmed as current by the syllabus index.",
            "B.Tech CSE first semester syllabus",
        )

        self.assertIn(BTECH_CSE_SYLLABUS_URL, answer)
        self.assertIn("effective from 2017-18", answer)
        self.assertIn("Year 2, Semester III", answer)
        self.assertIn("I won't guess", answer)
        self.assertNotIn("page 14", answer)
        self.assertNotIn("confirmed as current", answer)
        self.assertNotIn(SYLLABUS_INDEX_URL, answer)

    def test_first_semester_override_handles_common_hinglish_question(self) -> None:
        answer = _append_btech_cse_syllabus_pdf(
            "Unverified generated answer.",
            "B.Tech CSE ka 1 semester syllabus",
        )

        self.assertIn("Semester III", answer)
        self.assertIn("guess nahi karunga", answer)
        self.assertIn(BTECH_CSE_SYLLABUS_URL, answer)

    def test_syllabus_intent_handles_hindi_and_common_typo(self) -> None:
        self.assertTrue(_is_syllabus_request("सिलेबस बताइए"))
        self.assertTrue(_is_syllabus_request("BTech sllybuse"))
        self.assertFalse(_is_syllabus_request("Where is the library?"))

    def test_only_answer_cited_sources_are_linked(self) -> None:
        messages = [
            ToolMessage(
                content=(
                    "[BU Vice-Chancellor, p. 1] "
                    "(applicability: live directory)\n"
                    "Official source: https://www.bujhansi.ac.in/"
                    "administration/vice-chancellor/\n"
                    "Prof. Mukesh Pandey\n\n"
                    "[BU Hostel Fees] (applicability: current page)\n"
                    "Official source: https://www.bujhansi.ac.in/"
                    "student-life/hostels-facilities/hostel-fee/\n"
                    "Hostel fee details."
                ),
                tool_call_id="search-1",
            ),
        ]
        answer = "The Vice-Chancellor is Prof. Mukesh Pandey [BU Vice-Chancellor, p. 1]."

        self.assertEqual(
            get_cited_document_source_links(answer, messages),
            [
                "[BU Vice-Chancellor, p. 1]("
                "https://www.bujhansi.ac.in/administration/vice-chancellor/)"
            ],
        )

    def test_uncited_retrieved_documents_are_not_added_as_sources(self) -> None:
        messages = [
            ToolMessage(
                content=(
                    "[BU Placement Procedure] "
                    "(applicability: live procedure page)\n"
                    "Official source: https://ums.bujhansi.ac.in/"
                    "BUJhansi/Placement/placement-procedure.html\n"
                    "Students register for a placement drive."
                ),
                tool_call_id="search-1",
            ),
        ]

        self.assertEqual(
            get_cited_document_source_links(
                "The assistant could not confirm this detail.",
                messages,
            ),
            [],
        )

    def test_search_budget_forces_final_answer_after_one_search(self) -> None:
        messages = [
            ToolMessage(content="official BU result", tool_call_id="search-1"),
        ]

        self.assertTrue(_search_budget_exhausted(messages))
        self.assertFalse(_search_budget_exhausted([]))

    def test_cached_retriever_loads_without_rebuilding_embeddings(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            cache_directory = Path(temporary_directory)
            (cache_directory / "index.faiss").write_bytes(b"index")
            (cache_directory / "index.pkl").write_bytes(b"store")
            vector_store = MagicMock()

            with (
                patch(
                    "backend.agent.FAISS.load_local",
                    return_value=vector_store,
                ) as load_local,
                patch("backend.agent.FAISS.from_documents") as from_documents,
            ):
                retriever = build_retriever(
                    [],
                    MagicMock(),
                    cache_directory,
                )

        load_local.assert_called_once()
        from_documents.assert_not_called()
        vector_store.as_retriever.assert_called_once_with(
            search_kwargs={"k": 4}
        )
        self.assertIs(
            retriever,
            vector_store.as_retriever.return_value,
        )

    def test_changed_sources_update_previous_index_incrementally(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            cache_root = Path(temporary_directory)
            previous_cache = cache_root / "previous"
            previous_cache.mkdir()
            (previous_cache / "index.faiss").write_bytes(b"index")
            (previous_cache / "index.pkl").write_bytes(b"store")
            current_cache = cache_root / "current"
            unchanged_url = "https://www.bujhansi.ac.in/unchanged"
            changed_url = "https://www.bujhansi.ac.in/changed"
            old_documents = {
                "unchanged-id": Document(
                    page_content="Current unchanged official text.",
                    metadata={"source_url": unchanged_url},
                ),
                "changed-id": Document(
                    page_content="Old official text.",
                    metadata={"source_url": changed_url},
                ),
            }
            vector_store = MagicMock()
            vector_store.index_to_docstore_id = {
                "0": "unchanged-id",
                "1": "changed-id",
            }
            vector_store.docstore.search.side_effect = old_documents.get
            current_documents = [
                Document(
                    page_content="Current unchanged official text.",
                    metadata={"source_url": unchanged_url},
                ),
                Document(
                    page_content="Updated official text.",
                    metadata={"source_url": changed_url},
                ),
                Document(
                    page_content="New placement procedure.",
                    metadata={
                        "source_url": (
                            "https://ums.bujhansi.ac.in/placement"
                        )
                    },
                ),
            ]

            with (
                patch(
                    "backend.agent.FAISS.load_local",
                    return_value=vector_store,
                ) as load_local,
                patch("backend.agent.FAISS.from_documents") as from_documents,
            ):
                retriever = build_retriever(
                    current_documents,
                    MagicMock(),
                    current_cache,
                )

        load_local.assert_called_once_with(
            str(previous_cache),
            ANY,
            allow_dangerous_deserialization=True,
        )
        vector_store.delete.assert_called_once_with(ids=["changed-id"])
        added_documents = vector_store.add_documents.call_args.args[0]
        self.assertEqual(
            {document.page_content for document in added_documents},
            {"Updated official text.", "New placement procedure."},
        )
        vector_store.save_local.assert_called_once_with(
            str(current_cache)
        )
        from_documents.assert_not_called()
        self.assertIs(
            retriever,
            vector_store.as_retriever.return_value,
        )

    def test_source_fingerprint_changes_when_pdf_changes(self) -> None:
        with (
            TemporaryDirectory() as temporary_directory,
            patch(
                "backend.agent.UNIVERSITY_DATA_DIR",
                Path(temporary_directory),
            ),
            patch("backend.agent.UNIVERSITY_SOURCES", ()),
        ):
            syllabus_directory = Path(temporary_directory) / "syllabi"
            syllabus_directory.mkdir()
            pdf_path = syllabus_directory / "course.pdf"
            pdf_path.write_bytes(b"%PDF-original")
            pdf_path.with_suffix(".json").write_text(
                (
                    '{"title":"BU course syllabus",'
                    '"url":"https://www.bujhansi.ac.in/media/course.pdf",'
                    '"academic_year":"2026-27","coverage":"course"}'
                ),
                encoding="utf-8",
            )
            original_fingerprint = _university_source_fingerprint()

            pdf_path.write_bytes(b"%PDF-updated with different content")
            updated_fingerprint = _university_source_fingerprint()

        self.assertNotEqual(original_fingerprint, updated_fingerprint)

    def test_summarizes_bundelkhand_university_search(self) -> None:
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_bu_information",
                        "args": {"query": "2026-27 MCA fees"},
                        "id": "search-1",
                        "type": "tool_call",
                    },
                ],
            ),
            ToolMessage(
                content=(
                    "[BU Academic Bulletin and Updated Fees 2026-27, p. 4] "
                    "Official source: https://www.bujhansi.ac.in/"
                ),
                tool_call_id="search-1",
            ),
        ]

        actions = summarize_agent_actions(messages)

        self.assertEqual(
            actions,
            [
                "Searched Bundelkhand University sources for: "
                "2026-27 MCA fees — sources: "
                "[BU Academic Bulletin and Updated Fees 2026-27, p. 4]"
            ],
        )


if __name__ == "__main__":
    unittest.main()
