import hmac
import os
from datetime import date
from typing import Annotated, Literal

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
    status,
)
from groq import RateLimitError
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage

from backend.agent import get_agent, get_btech_cse_first_semester_answer
from backend.config import load_project_environment
from backend.documents import (
    MAX_PDF_BYTES,
    answer_pdf_question,
    get_document_llm,
    get_pdf_content_and_retriever,
    get_pdf_pages,
    summarize_pdf,
)
from backend.study_planner import create_study_plan
from backend.university_sources import get_university_source_status


load_project_environment()

app = FastAPI(
    title="AI-Powered Student Assistant API",
    description=(
        "Generative AI and agentic RAG backend for the AI-Powered Student Assistant."
    ),
    version="1.0.0",
)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)


class ChatRequest(BaseModel):
    department: str = Field(default="", max_length=160)
    messages: list[ChatTurn] = Field(min_length=1, max_length=30)


class ChatResponse(BaseModel):
    answer: str
    category: Literal["academic", "fee", "general"]
    agent_actions: list[str]


class DocumentAnalysisResponse(BaseModel):
    filename: str
    mode: Literal["summary", "question"]
    answer: str
    citations: list[str]


class StudyPlanActivityResponse(BaseModel):
    topic: str
    task: str
    category: Literal["learn", "practice", "revision"]
    minutes: int
    citations: list[str]


class StudyPlanDayResponse(BaseModel):
    date: date
    focus: str
    activities: list[StudyPlanActivityResponse]
    planned_minutes: int


class StudyPlanResponse(BaseModel):
    filename: str
    title: str
    overview: str
    exam_date: date
    daily_hours: float
    days: list[StudyPlanDayResponse]
    citations: list[str]


class UniversitySourceResponse(BaseModel):
    title: str
    url: str
    kind: Literal["pdf", "webpage"]
    academic_year: str
    coverage: str
    available: bool
    last_fetched: str | None
    last_modified: str | None


class UniversitySourcesResponse(BaseModel):
    university: str
    available_count: int
    sources: list[UniversitySourceResponse]


def require_api_key(
    api_key: Annotated[str | None, Header(alias="X-CampusGuide-Key")] = None,
) -> None:
    expected_key = os.getenv("CAMPUSGUIDE_API_KEY")
    if not expected_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The API server has not been configured.",
        )
    if api_key is None or not hmac.compare_digest(api_key, expected_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )


@app.get("/health")
def health() -> dict[str, str | list[str]]:
    return {
        "status": "ok",
        "features": [
            "chat",
            "documents",
            "study_planner",
            "bundelkhand_university_sources",
        ],
    }


@app.post(
    "/api/chat",
    response_model=ChatResponse,
    dependencies=[Depends(require_api_key)],
)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        first_semester_answer = get_btech_cse_first_semester_answer(
            request.messages[-1].content
        )
        if first_semester_answer is not None:
            return ChatResponse(
                answer=first_semester_answer,
                category="academic",
                agent_actions=[],
            )
        conversation = [
            HumanMessage(content=turn.content)
            if turn.role == "user"
            else AIMessage(content=turn.content)
            for turn in request.messages
        ]
        result = get_agent().answer(
            department=request.department,
            conversation=conversation,
        )
        return ChatResponse(**result)
    except RateLimitError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The AI provider is temporarily rate-limited. Wait a minute "
                "and try again; your message and university sources are safe."
            ),
        ) from error
    except RuntimeError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error


@app.get(
    "/api/university/sources",
    response_model=UniversitySourcesResponse,
    dependencies=[Depends(require_api_key)],
)
def list_university_sources() -> UniversitySourcesResponse:
    sources = get_university_source_status()
    return UniversitySourcesResponse(
        university="Bundelkhand University, Jhansi",
        available_count=sum(
            1 for source in sources if source["available"]
        ),
        sources=sources,
    )


@app.post(
    "/api/university/reload",
    dependencies=[Depends(require_api_key)],
)
def reload_university_search_index() -> dict[str, str]:
    get_agent.cache_clear()
    return {
        "status": "ok",
        "message": (
            "The current search index was cleared and will rebuild from "
            "the official Bundelkhand University source pack on the next chat."
        ),
    }


@app.post(
    "/api/documents/analyze",
    response_model=DocumentAnalysisResponse,
    dependencies=[Depends(require_api_key)],
)
def analyze_uploaded_document(
    file: UploadFile = File(...),
    mode: Literal["summary", "question"] = Form(...),
    question: str = Form(default="", max_length=2000),
) -> DocumentAnalysisResponse:
    content = file.file.read(MAX_PDF_BYTES + 1)
    if mode == "question" and not question.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Enter a question about the uploaded PDF.",
        )

    try:
        llm = get_document_llm()
    except RuntimeError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error

    try:
        if mode == "summary":
            pages = get_pdf_pages(file.filename or "", content)
            answer, citations = summarize_pdf(llm, pages)
        else:
            pages, retriever = get_pdf_content_and_retriever(
                file.filename or "",
                content,
            )
            answer, citations = answer_pdf_question(
                llm,
                retriever,
                question.strip(),
            )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error

    return DocumentAnalysisResponse(
        filename=file.filename or "uploaded.pdf",
        mode=mode,
        answer=answer,
        citations=citations,
    )


@app.post(
    "/api/documents/study-plan",
    response_model=StudyPlanResponse,
    dependencies=[Depends(require_api_key)],
)
def create_uploaded_syllabus_study_plan(
    file: UploadFile = File(...),
    exam_date: date = Form(...),
    daily_hours: float = Form(..., gt=0, le=12),
) -> StudyPlanResponse:
    try:
        pages = get_pdf_pages(
            file.filename or "",
            file.file.read(MAX_PDF_BYTES + 1),
        )
        llm = get_document_llm()
        plan = create_study_plan(
            llm=llm,
            pages=pages,
            exam_date=exam_date,
            daily_hours=daily_hours,
        )
    except RuntimeError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error

    return StudyPlanResponse(
        filename=file.filename or "uploaded.pdf",
        **plan,
    )
