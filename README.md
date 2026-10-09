# Bundelkhand University AI-Powered Student Assistant

A student-facing assistant for Bundelkhand University, Jhansi. It combines a
LangGraph tool-calling agent, retrieval-augmented generation (RAG), official
university sources, PDF Q&A, and a syllabus study planner.

## Current official-source coverage

The source pack is maintained from the official `bujhansi.ac.in` university
site and currently catalogs 52 official pages and documents, plus PDFs
discovered from the live syllabus and examination pages:

- Academic bulletin and updated admission/examination fees for 2026–27.
- Admission information and guidelines for 2026.
- Official department-wise syllabus index and linked syllabus PDFs across
  undergraduate, postgraduate, engineering, management and other programmes,
  including B.Tech, BCA, MCA and MBA where published by BU.
- Vice-Chancellor, Registrar, Finance Officer, Controller of Examinations,
  Deans/Heads of Departments, campus faculty and other official administration
  directories.
- PhD admission notice and guidelines for 2026–27.
- Three- and four-year UG programme ordinances and fee-refund policy.
- Hostel charges, hostel rules, disability policy, and hostel facilities.
- Live admissions and syllabus index pages.
- Examination schemes and examination notices.
- Examination formats, fee-refund and general notices.
- Anti-ragging, health, equal-opportunity, SEDG, cyber-safety, and student
  clubs information.
- FYUP professional/traditional course pages and affiliated-college directory.
- Student results, central library, e-content, NSS and NCC pages.
- University contact and institutes/departments directories.

Every retrieved passage carries its official URL, source title, and academic
year/applicability label. PDF answers can cite page numbers. The assistant must
say when indexed documents do not establish an answer; it must not guess.
Fees, schedules, eligibility and regulations should be checked against the
linked official source before a student acts.

The live syllabus index and examination scheme/notice tables are refreshed as
part of the full source refresh. Only official BU HTTPS PDF links are
downloaded; each syllabus retains its department/programme title. The official
index does not publish a session year for every syllabus, so the assistant
flags applicability when BU does not state one. BU-specific scholarship rules
were not found in the current official source set. If a document changes or
the university publishes a correction, refresh the source pack and verify the
affected answers.
Some indexed rule documents have an unclear or older effective session
(including the 2021 hostel rulebook). The assistant labels that uncertainty;
students should confirm the current rule with the university before acting.
The hostel charge PDF is a scanned memo dated 18 July 2019. Its searchable
transcription is tied to the downloaded PDF checksum and is labeled historical;
do not treat those rates as confirmed current fees. Scanned first-year pages
in the B.Tech CSE syllabus are OCR-indexed in the Docker image; verify the
recognized text against the original PDF.

## Run locally

1. Create and activate a Python virtual environment.
2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

   Install Tesseract OCR and make `tesseract` available on `PATH`; the
   assistant uses it to index the scanned pages of the B.Tech CSE PDF. Docker
   installs the OCR engine automatically.

3. Configure `.env` using `.env.example`. Set `GROQ_API_KEY` and
   `CAMPUSGUIDE_API_KEY` (the latter is a legacy configuration name retained
   so existing local setups continue to work). Use a long random value for the
   API key; it is used only for server-to-server requests.
4. Download the university source pack (it is intentionally not stored in
   GitHub because it is a large collection of PDFs and generated files):

   ```powershell
   python -m backend.refresh_university_sources
   ```

   The downloaded files and local search index are kept under `data/` and are
   ignored by Git.
5. Start the app from this project folder:

   ```powershell
   python -m streamlit run bundelkhand_student_assistant.py
   ```

The Streamlit app starts a local FastAPI backend when needed. The API is
available at `http://127.0.0.1:8000/docs`. For a separately managed backend:

```powershell
python -m uvicorn backend.main:app --reload --port 8000
```

## Run with Docker

Install and start Docker Desktop, then run these commands from the project
folder. Copy `.env.example` to `.env` and add your API keys, then download the
university source pack with `python -m backend.refresh_university_sources`.
Compose starts the Streamlit frontend and its local FastAPI backend together
in one container:

```powershell
docker compose up --build
```

Open `http://localhost:8501`. The API key is read from `.env` on the host
and is not copied into the image. University source files and the local search
index remain in `data/bundelkhand_university`; the Hugging Face model cache is
kept in a Docker volume. The first search after an index change can take
several minutes; later starts reuse the saved index.

Stop the service with `Ctrl+C`, or run `docker compose down` from the project
folder. Docker only makes the app available on this computer. A public,
shareable internet link requires deploying the container to a hosting service
and configuring HTTPS and server-side secrets.

The first chat builds a local sentence-transformer embedding model and search
index; it may take several minutes and needs internet access the first time to
download the model. The index is cached locally and reused until an indexed
source changes. Keep API keys in server-side environment variables or secrets.

## Refresh official sources

Refresh all allowlisted university PDFs and pages from official BU hosts:

```powershell
python -m backend.refresh_university_sources
```

Refresh only selected source files when a particular official document
changes:

```powershell
python -m backend.refresh_university_sources --source hostel_fees --source hostel_charges
```

Refresh only the linked syllabus and exam PDFs without refreshing the complete
static source pack:

```powershell
python -m backend.refresh_university_sources --source syllabus_index --source examination_schemes --source examination_notices --linked
```

The downloader accepts only HTTPS URLs in the source catalog and rejects
redirects outside official BU hosts. On a refresh failure, the old downloaded
copy remains available and the command reports the failure. Each file has a
small metadata sidecar with its official URL, fetch time, Last-Modified value
when supplied, and SHA-256 digest. Linked PDFs are streamed and limited to
50 MB each; the linked-document cache is capped at 1 GB. Larger files are
reported as unavailable rather than silently omitted. Do not present
session-sensitive material as current if its academic year is unclear.
The source inventory checks file and metadata presence without reading every
PDF; PDF SHA-256 checks are performed when building the search index.

The app's **Official university knowledge base** section lists which source
files are present, their coverage, applicability, and source links.

## Features

- **University Q&A:** the agent searches across the indexed official BU source
  pack, can refine its search, answers in the student's language, and shows
  source searches, citations, and official URLs. Syllabus questions always
  include BU's official programme-wise syllabus index as a fallback; students
  should confirm the programme and session on the university page.
- **B.Tech CSE syllabus PDF:** matching syllabus questions include a direct
  download of BU's linked PDF. That file says its scheme is effective from
  2017-18, so the assistant warns that current-session applicability is not
  confirmed. The scanned first-year pages are OCR-indexed; verify recognized
  course codes and details against the original PDF.
- **Training and placement:** official BU placement-cell, student-guidance,
  and placement-procedure pages are indexed so the assistant can describe the
  process in chat. Placement records without a stated reporting period and
  historical drive notices are identified as such, not presented as current
  B.Tech outcomes.
- **Fee/admission/exam/policy information:** answers are grounded in the
  relevant official passages and carry their session/year. There is
  intentionally no hard-coded or fictional fee calculator.
- **Upload and ask:** students can summarize or ask questions about their own
  text-based PDF; answers cite its page numbers.
- **Syllabus study planner:** a student uploads a syllabus, enters the exam
  date and available daily hours, and receives a cited learning/practice/
  revision schedule with session-only progress checkboxes.
- **FastAPI endpoints:** authenticated chat, source inventory, PDF analysis,
  study-plan generation, and search-index reload are defined in
  `backend/main.py`.

Uploaded document text is sent to the configured Groq model and held
temporarily in session/backend caches. Do not upload confidential student
records. Scanned/image-only PDFs need OCR before upload. The app has no student
login or permanent progress storage.

## Project structure

- `bundelkhand_student_assistant.py` — Bundelkhand University Streamlit app.
- `backend/agent.py` — LangGraph agent and official-source retrieval.
- `backend/university_sources.py` — official source catalog, loader, status and
  safe downloader.
- `backend/refresh_university_sources.py` — source refresh command.
- `backend/documents.py` — uploaded-PDF RAG and summaries.
- `backend/study_planner.py` — cited syllabus-to-study planning.
- `data/bundelkhand_university/` — official-source PDFs, page snapshots and
  fetch metadata.
- `tests/` — source, retrieval, document Q&A, study-plan, and agent tests.

## Run tests

From the project folder:

```powershell
python -m unittest discover -s tests -v
```

Automated tests use fake language models for deterministic behavior. They do
not certify live model-answer quality or make the assistant an official
university service.
