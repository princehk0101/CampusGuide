import hashlib
import io
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch
from urllib.parse import urlparse

import pymupdf
import requests
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from backend.agent import BTECH_CSE_SYLLABUS_URL
from backend.main import app
from backend.university_sources import (
    OFFICIAL_HOSTS,
    UNIVERSITY_SOURCES,
    UNIVERSITY_DATA_DIR,
    _PageTextExtractor,
    _assert_official_url,
    get_university_source_status,
    load_university_documents,
    _linked_pdf_records,
    refresh_university_sources,
    refresh_linked_university_documents,
)


class UniversitySourceTests(unittest.TestCase):
    def test_source_catalog_uses_only_official_bu_hosts(self) -> None:
        for source in UNIVERSITY_SOURCES:
            self.assertEqual(urlparse(source.url).scheme, "https")
            self.assertIn(urlparse(source.url).hostname, OFFICIAL_HOSTS)

    def test_official_placement_procedure_is_searchable(self) -> None:
        source = next(
            source
            for source in UNIVERSITY_SOURCES
            if source.key == "training_placement_procedure"
        )
        with TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            (data_dir / source.filename).write_text(
                (
                    "Interested students register before the deadline. "
                    "The company may shortlist applicants and conduct "
                    "a test, group discussion and interview."
                ),
                encoding="utf-8",
            )
            documents = load_university_documents(data_dir)

        procedure = next(
            document
            for document in documents
            if document.metadata.get("source_key")
            == "training_placement_procedure"
        )
        self.assertEqual(procedure.metadata["source_url"], source.url)
        self.assertIn("register before the deadline", procedure.page_content)

    def test_incremental_loading_reads_only_sources_newer_than_cache(self) -> None:
        placement_sources = {
            source.key: source
            for source in UNIVERSITY_SOURCES
            if source.key in {
                "training_placement",
                "training_placement_procedure",
            }
        }
        with TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            old_source_path = data_dir / placement_sources[
                "training_placement"
            ].filename
            old_source_path.write_text(
                "Unchanged placement-cell overview.",
                encoding="utf-8",
            )
            old_source_path.touch()
            modified_after = old_source_path.stat().st_mtime + 1
            updated_source_path = data_dir / placement_sources[
                "training_placement_procedure"
            ].filename
            updated_source_path.write_text(
                "Updated placement procedure.",
                encoding="utf-8",
            )
            os.utime(
                updated_source_path,
                (modified_after + 10, modified_after + 10),
            )

            documents = load_university_documents(
                data_dir,
                modified_after=modified_after,
            )

        self.assertEqual(len(documents), 1)
        self.assertEqual(
            documents[0].metadata["source_key"],
            "training_placement_procedure",
        )

    def test_official_campus_faculty_directory_is_indexed(self) -> None:
        source = next(
            source
            for source in UNIVERSITY_SOURCES
            if source.key == "campus_faculty"
        )
        with TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            (data_dir / source.filename).write_text(
                "Department of Hindi: Dr Puneet Bisaria (HOD).",
                encoding="utf-8",
            )
            documents = load_university_documents(data_dir)

        self.assertEqual(len(documents), 1)
        self.assertEqual(
            documents[0].metadata["source_url"],
            source.url,
        )
        self.assertIn("Puneet Bisaria", documents[0].page_content)

    def test_official_hostel_charge_pdf_is_indexed(self) -> None:
        source = next(
            source
            for source in UNIVERSITY_SOURCES
            if source.key == "hostel_charges"
        )

        self.assertEqual(
            source.url,
            "https://bujhansi.ac.in/media/ulafdkj1/638715099957836008.pdf",
        )
        with TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            for filename in (
                "hostel_fees.txt",
                "hostel_fees.txt.json",
                "hostel_charges.pdf",
                "hostel_charges.pdf.json",
                "hostel_charges.ocr.txt",
                "hostel_charges.ocr.txt.json",
            ):
                (data_dir / filename).write_bytes(
                    (UNIVERSITY_DATA_DIR / filename).read_bytes()
                )
            documents = load_university_documents(data_dir)
        hostel_pages = [
            document
            for document in documents
            if document.metadata.get("source_key") == "hostel_charges"
        ]
        self.assertTrue(hostel_pages)
        self.assertTrue(
            all(
                document.metadata["source_url"] == source.url
                and isinstance(document.metadata.get("page"), int)
                for document in hostel_pages
            )
        )
        self.assertIn("10,000", hostel_pages[0].page_content)
        self.assertIn("8,200", hostel_pages[0].page_content)
        self.assertIn("18 July 2019", hostel_pages[0].page_content)

    def test_scanned_btech_cse_pages_are_ocr_indexed(self) -> None:
        source = next(
            source
            for source in UNIVERSITY_SOURCES
            if source.key == "btech_cse_syllabus"
        )
        with TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            source_path = data_dir / source.filename
            source_path.write_bytes(b"scanned syllabus")
            fake_pdf_page = MagicMock()
            fake_pdf_page.get_textpage_ocr.return_value = object()
            fake_pdf_page.get_text.return_value = (
                "Semester I: Engineering Mathematics"
            )
            fake_pdf = MagicMock()
            fake_pdf.__getitem__.return_value = fake_pdf_page
            fake_pdf.__enter__.return_value = fake_pdf
            with (
                patch(
                    "backend.university_sources.PyPDFLoader"
                ) as pdf_loader,
                patch(
                    "backend.university_sources.pymupdf.open",
                    return_value=fake_pdf,
                ),
            ):
                pdf_loader.return_value.load.return_value = [
                    Document(
                        page_content="",
                        metadata={"page": 0},
                    )
                ]
                documents = load_university_documents(
                    data_dir,
                    modified_after=source_path.stat().st_mtime + 100,
                    force_source_keys={source.key},
                )

        self.assertEqual(len(documents), 1)
        self.assertIn("Semester I", documents[0].page_content)
        self.assertEqual(documents[0].metadata["page"], 0)

    def test_scanned_pdf_transcription_is_ignored_after_source_changes(self) -> None:
        source = next(
            source
            for source in UNIVERSITY_SOURCES
            if source.key == "hostel_charges"
        )
        with TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            (data_dir / source.filename).write_bytes(
                (UNIVERSITY_DATA_DIR / source.filename).read_bytes()
            )
            (data_dir / "hostel_fees.txt").write_text(
                "The current hostel fee page links to official charges.",
                encoding="utf-8",
            )
            (data_dir / "hostel_charges.ocr.txt").write_text(
                "Stale transcription",
                encoding="utf-8",
            )
            (data_dir / "hostel_charges.ocr.txt.json").write_text(
                '{"source_sha256": "old-source-hash"}',
                encoding="utf-8",
            )
            (data_dir / "hostel_charges.pdf.json").write_text(
                '{"sha256": "new-source-hash"}',
                encoding="utf-8",
            )

            documents = load_university_documents(data_dir)

        self.assertFalse(
            any(
                document.metadata.get("source_key") == "hostel_charges"
                for document in documents
            )
        )

    def test_source_fetcher_rejects_external_hosts(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-official"):
            _assert_official_url("https://example.com/university-fees.pdf")

    def test_html_source_extractor_keeps_only_official_links(self) -> None:
        extractor = _PageTextExtractor()
        extractor.feed(
            '<html><script>secret();</script><body><h1>Admissions</h1>'
            '<a href="/media/prospectus.pdf">Prospectus</a>'
            '<a href="https://example.com">External</a></body></html>'
        )

        text = extractor.to_text("https://www.bujhansi.ac.in/admissions/")

        self.assertIn("Admissions", text)
        self.assertIn(
            "https://www.bujhansi.ac.in/media/prospectus.pdf",
            text,
        )
        self.assertNotIn("secret()", text)
        self.assertNotIn("example.com", text)

    def test_syllabus_link_preserves_department_and_programme(self) -> None:
        response = requests.Response()
        response.status_code = 200
        response.url = "https://www.bujhansi.ac.in/academics/syllabus/"
        response._content = (
            b"<table><tr><td>1</td><td>Institute of Computing</td>"
            b"<td>B.Tech Computer Science</td><td><a href='/media/test.pdf'>"
            b"Click Here</a></td></tr></table>"
        )
        with patch(
            "backend.university_sources.requests.get",
            return_value=response,
        ):
            records = _linked_pdf_records(
                response.url,
                "syllabus",
            )

        self.assertEqual(len(records), 1)
        self.assertEqual(
            records[0]["title"],
            "BU Syllabus — B.Tech Computer Science — Institute of Computing",
        )
        self.assertEqual(
            records[0]["url"],
            "https://www.bujhansi.ac.in/media/test.pdf",
        )

    def test_linked_syllabus_documents_are_loaded_with_official_citations(
        self,
    ) -> None:
        source_pdf = (
            UNIVERSITY_DATA_DIR / "bca_syllabus.pdf"
        ).read_bytes()
        digest = hashlib.sha256(source_pdf).hexdigest()
        with TemporaryDirectory() as temporary_directory:
            syllabus_directory = Path(temporary_directory) / "syllabi"
            syllabus_directory.mkdir()
            pdf_path = syllabus_directory / "bca-syllabus.pdf"
            pdf_path.write_bytes(source_pdf)
            pdf_path.with_suffix(".json").write_text(
                (
                    '{"title":"BU Syllabus — BCA — Institute of Computing",'
                    '"url":"https://www.bujhansi.ac.in/media/tmdezlty/bca.pdf",'
                    '"academic_year":"Live official syllabus index",'
                    '"coverage":"BCA syllabus",'
                    f'"sha256":"{digest}"' + "}"
                ),
                encoding="utf-8",
            )
            documents = load_university_documents(Path(temporary_directory))

        linked_documents = [
            document
            for document in documents
            if document.metadata.get("source_key")
            == "syllabi:bca-syllabus"
        ]
        self.assertTrue(linked_documents)
        self.assertTrue(
            all(
                document.metadata["source_url"]
                == "https://www.bujhansi.ac.in/media/tmdezlty/bca.pdf"
                for document in linked_documents
            )
        )

    def test_loaded_pages_keep_official_source_and_session_metadata(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            bulletin_source = UNIVERSITY_DATA_DIR / "admission_bulletin_2026_27.pdf"
            (data_dir / bulletin_source.name).write_bytes(
                bulletin_source.read_bytes()
            )
            bulletin_metadata = bulletin_source.with_suffix(".pdf.json")
            (data_dir / bulletin_metadata.name).write_bytes(
                bulletin_metadata.read_bytes()
            )
            documents = load_university_documents(data_dir)

        self.assertTrue(documents)
        bulletin_pages = [
            page
            for page in documents
            if page.metadata.get("source_key")
            == "admission_bulletin_2026_27"
        ]
        self.assertTrue(bulletin_pages)
        self.assertTrue(
            all(
                page.metadata["academic_year"] == "2026-27"
                and page.metadata["source_url"].startswith(
                    "https://www.bujhansi.ac.in/"
                )
                and isinstance(page.metadata.get("page"), int)
                for page in bulletin_pages
            )
        )

    def test_missing_legacy_or_non_catalog_files_are_not_indexed(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            (data_dir / "academics_handbook.pdf").write_bytes(b"%PDF-fake")
            (data_dir / "fee_structure.pdf").write_bytes(b"%PDF-fake")

            with self.assertRaisesRegex(FileNotFoundError, "source pack"):
                load_university_documents(data_dir)

    def test_source_status_marks_unavailable_files(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            status = get_university_source_status(
                Path(temporary_directory)
            )

        self.assertEqual(len(status), len(UNIVERSITY_SOURCES))
        self.assertTrue(all(not source["available"] for source in status))

    def test_linked_source_status_does_not_read_large_pdf_contents(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            syllabus_directory = data_dir / "syllabi"
            syllabus_directory.mkdir()
            metadata = {
                "title": "BU B.Tech Syllabus",
                "url": "https://www.bujhansi.ac.in/media/btech.pdf",
                "academic_year": "2026-27",
                "coverage": "B.Tech syllabus",
                "sha256": "a" * 64,
            }
            (syllabus_directory / "btech.pdf").write_bytes(b"%PDF-test")
            (syllabus_directory / "btech.json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )

            with patch.object(
                Path,
                "read_bytes",
                side_effect=AssertionError(
                    "source status must not read PDF contents"
                ),
            ):
                status = get_university_source_status(data_dir)

        linked_source = next(
            source for source in status if source["title"] == metadata["title"]
        )
        self.assertTrue(linked_source["available"])

    def test_refresh_can_update_only_selected_sources(self) -> None:
        response = requests.Response()
        response.status_code = 200
        response.url = (
            "https://bujhansi.ac.in/media/ulafdkj1/"
            "638715099957836008.pdf"
        )
        response._content = b"%PDF-test"
        response.headers["Last-Modified"] = "Wed, 07 Oct 2026 00:00:00 GMT"

        with TemporaryDirectory() as temporary_directory:
            with patch(
                "backend.university_sources.requests.get",
                return_value=response,
            ) as get:
                results = refresh_university_sources(
                    Path(temporary_directory),
                    source_keys={"hostel_charges"},
                )

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["status"], "updated")
            get.assert_called_once()
            self.assertTrue(
                (
                    Path(temporary_directory)
                    / "hostel_charges.pdf"
                ).is_file()
            )
            self.assertFalse(
                (
                    Path(temporary_directory)
                    / "admission_bulletin_2026_27.pdf"
                ).exists()
            )

    def test_linked_refresh_streams_large_pdf_with_dedicated_limit(self) -> None:
        record = {
            "url": "https://www.bujhansi.ac.in/media/large-syllabus.pdf",
            "title": "BU large syllabus",
            "kind": "syllabus",
            "academic_year": "2026-27",
            "coverage": "large syllabus",
        }
        response = requests.Response()
        response.status_code = 200
        response.url = record["url"]
        response.raw = io.BytesIO()
        response.headers["Content-Length"] = "9"
        response.iter_content = lambda **kwargs: iter(
            [b"%PDF-test"] if kwargs["chunk_size"] == 64 * 1024 else []
        )

        with (
            TemporaryDirectory() as temporary_directory,
            patch(
                "backend.university_sources._linked_pdf_records",
                return_value=[record],
            ),
            patch(
                "backend.university_sources.requests.get",
                return_value=response,
            ) as get,
            patch("backend.university_sources.MAX_SOURCE_BYTES", 8),
            patch("backend.university_sources.MAX_LINKED_SOURCE_BYTES", 16),
        ):
            results = refresh_linked_university_documents(
                Path(temporary_directory)
            )

            self.assertEqual(
                sum(result["status"] == "updated" for result in results),
                1,
            )
            get.assert_called_once()
            self.assertTrue(get.call_args.kwargs["stream"])
            downloaded_pdf = next(
                (Path(temporary_directory) / "syllabi").glob("*.pdf")
            )
            self.assertEqual(downloaded_pdf.read_bytes(), b"%PDF-test")
            self.assertTrue(downloaded_pdf.with_suffix(".json").is_file())

    def test_linked_refresh_rejects_pdf_above_its_per_file_limit(self) -> None:
        record = {
            "url": "https://www.bujhansi.ac.in/media/large-syllabus.pdf",
            "title": "BU oversized syllabus",
            "kind": "syllabus",
            "academic_year": "2026-27",
            "coverage": "large syllabus",
        }
        response = requests.Response()
        response.status_code = 200
        response.url = record["url"]
        response.raw = io.BytesIO()
        response.headers["Content-Length"] = "9"
        response.iter_content = lambda **kwargs: iter(
            [] if kwargs["chunk_size"] == 64 * 1024 else [b"%PDF-test"]
        )

        with (
            TemporaryDirectory() as temporary_directory,
            patch(
                "backend.university_sources._linked_pdf_records",
                return_value=[record],
            ),
            patch(
                "backend.university_sources.requests.get",
                return_value=response,
            ),
            patch("backend.university_sources.MAX_LINKED_SOURCE_BYTES", 8),
        ):
            results = refresh_linked_university_documents(
                Path(temporary_directory)
            )

        self.assertIn("50 MB", results[0]["status"])
        self.assertFalse(
            list((Path(temporary_directory) / "syllabi").glob("*.pdf"))
        )

    def test_source_inventory_and_index_reload_require_api_key(self) -> None:
        os.environ["CAMPUSGUIDE_API_KEY"] = "bu-sources-test-key"
        with patch("backend.main.get_agent") as get_agent:
            with TestClient(app) as client:
                denied = client.get("/api/university/sources")
                response = client.get(
                    "/api/university/sources",
                    headers={"X-CampusGuide-Key": "bu-sources-test-key"},
                )
                reload_response = client.post(
                    "/api/university/reload",
                    headers={"X-CampusGuide-Key": "bu-sources-test-key"},
                )

        self.assertEqual(denied.status_code, 401)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            response.json()["university"],
            "Bundelkhand University, Jhansi",
        )
        available_files = get_university_source_status()
        self.assertEqual(
            response.json()["available_count"],
            sum(bool(source["available"]) for source in available_files),
        )
        self.assertGreaterEqual(
            len(response.json()["sources"]),
            len(UNIVERSITY_SOURCES),
        )
        self.assertEqual(reload_response.status_code, 200)
        get_agent.cache_clear.assert_called_once_with()

    def test_btech_first_semester_answer_skips_slow_agent_initialization(self) -> None:
        os.environ["CAMPUSGUIDE_API_KEY"] = "bu-chat-test-key"
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": "B.Tech CSE first semester syllabus",
                }
            ]
        }
        with patch("backend.main.get_agent") as get_agent:
            with TestClient(app) as client:
                response = client.post(
                    "/api/chat",
                    headers={"X-CampusGuide-Key": "bu-chat-test-key"},
                    json=payload,
                )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("Year 2, Semester III", response.json()["answer"])
        self.assertIn(BTECH_CSE_SYLLABUS_URL, response.json()["answer"])
        get_agent.assert_not_called()


if __name__ == "__main__":
    unittest.main()
