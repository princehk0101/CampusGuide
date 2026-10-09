# CampusGuide

AI student assistant for Bundelkhand University. Ask questions about university
documents, upload a PDF, or build a study plan from a syllabus.

## Tech stack

| Area | Technology |
| --- | --- |
| UI | Streamlit |
| API | FastAPI, Uvicorn |
| Agent | LangGraph, LangChain |
| LLM | Groq (`openai/gpt-oss-20b`) |
| RAG and search | Sentence Transformers (`all-MiniLM-L6-v2`), FAISS |
| PDF and OCR | PyMuPDF, pypdf, Tesseract OCR |
| Language | Python 3.13 |
| Container | Docker, Docker Compose |
| Tests | Python `unittest` |

The university document collection and FAISS index are stored locally under
`data/`; they are not included in GitHub.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Add `GROQ_API_KEY` and `CAMPUSGUIDE_API_KEY` to `.env`. Download the university
documents, then start the app:

```powershell
python -m backend.refresh_university_sources
python -m streamlit run bundelkhand_student_assistant.py
```

Open <http://localhost:8501>. Local OCR requires Tesseract on `PATH`.

## Run with Docker

Create `.env` from `.env.example`, add the two API keys, and download the source
documents with the command above. Then run:

```powershell
docker compose up --build
```

Open <http://localhost:8501>. The Docker image installs Tesseract OCR.

## Project layout

- `bundelkhand_student_assistant.py` — Streamlit interface
- `backend/agent.py` — LangGraph agent and retrieval
- `backend/main.py` — FastAPI endpoints
- `backend/university_sources.py` — official source catalog and document loader
- `backend/study_planner.py` — syllabus-based study plans
- `tests/` — automated tests

Run tests with `python -m unittest discover -s tests -v`.
