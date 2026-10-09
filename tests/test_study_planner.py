from datetime import date, timedelta
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.documents import Document

from backend.main import app
from backend.study_planner import (
    StudyPlanDraft,
    create_study_plan,
)


class FakeStructuredChatModel:
    def __init__(self, result: StudyPlanDraft) -> None:
        self.result = result

    def with_structured_output(self, schema: type[StudyPlanDraft]):
        self.asserted_schema = schema
        return self

    def invoke(self, messages: list) -> StudyPlanDraft:
        self.messages = messages
        return self.result


def sample_draft() -> StudyPlanDraft:
    return StudyPlanDraft.model_validate(
        {
            "title": "Exam preparation plan",
            "overview": "Study the syllabus, practise, and revise.",
            "activities": [
                {
                    "topic": "Algebra",
                    "task": "Review core concepts and worked examples.",
                    "category": "learn",
                    "minutes": 45,
                    "source_pages": [1],
                },
                {
                    "topic": "Algebra",
                    "task": "Solve representative problems.",
                    "category": "practice",
                    "minutes": 45,
                    "source_pages": [1],
                },
            ],
        }
    )


class StudyPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pages = (
            Document(
                page_content="Unit 1: Algebra and equations.",
                metadata={"page": 0},
            ),
            Document(
                page_content="Unit 2: Geometry.",
                metadata={"page": 1},
            ),
        )

    def test_distributes_activities_within_daily_time_and_cites_pages(self):
        model = FakeStructuredChatModel(sample_draft())

        plan = create_study_plan(
            model,
            self.pages,
            exam_date=date(2026, 1, 3),
            daily_hours=1,
            today=date(2026, 1, 1),
        )

        self.assertEqual(
            [day["date"] for day in plan["days"]],
            [
                date(2026, 1, 1),
                date(2026, 1, 2),
                date(2026, 1, 3),
            ],
        )
        self.assertEqual(
            [day["planned_minutes"] for day in plan["days"]],
            [45, 45, 0],
        )
        self.assertEqual(
            plan["days"][0]["activities"][0]["citations"],
            ["[Uploaded PDF, p. 1]"],
        )
        self.assertEqual(plan["citations"], ["[Uploaded PDF, p. 1]"])
        self.assertEqual(
            plan["days"][2]["focus"],
            "Revision / catch-up buffer",
        )

    def test_rejects_model_citation_outside_uploaded_pdf(self):
        draft = sample_draft()
        draft.activities[0].source_pages = [3]

        with self.assertRaisesRegex(ValueError, "not present"):
            create_study_plan(
                FakeStructuredChatModel(draft),
                self.pages,
                exam_date=date(2026, 1, 1),
                daily_hours=1,
                today=date(2026, 1, 1),
            )

    def test_rejects_plan_that_exceeds_available_days(self):
        with self.assertRaisesRegex(ValueError, "exceeds the time available"):
            create_study_plan(
                FakeStructuredChatModel(sample_draft()),
                self.pages,
                exam_date=date(2026, 1, 1),
                daily_hours=1,
                today=date(2026, 1, 1),
            )

    def test_study_plan_api_returns_session_ready_schedule(self):
        os.environ["CAMPUSGUIDE_API_KEY"] = "study-plan-test-key"
        with (
            patch(
                "backend.main.get_document_llm",
                return_value=FakeStructuredChatModel(sample_draft()),
            ),
            patch("backend.main.get_pdf_pages", return_value=self.pages),
        ):
            with TestClient(app) as client:
                request_files = {
                    "file": (
                        "syllabus.pdf",
                        b"%PDF-test",
                        "application/pdf",
                    ),
                }
                request_data = {
                    "exam_date": (
                        date.today() + timedelta(days=2)
                    ).isoformat(),
                    "daily_hours": "1",
                }
                denied = client.post(
                    "/api/documents/study-plan",
                    files=request_files,
                    data=request_data,
                )
                response = client.post(
                    "/api/documents/study-plan",
                    headers={"X-CampusGuide-Key": "study-plan-test-key"},
                    files=request_files,
                    data=request_data,
                )

        self.assertEqual(denied.status_code, 401)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["filename"], "syllabus.pdf")
        self.assertEqual(len(response.json()["days"]), 3)
        self.assertEqual(response.json()["days"][0]["planned_minutes"], 45)


if __name__ == "__main__":
    unittest.main()
