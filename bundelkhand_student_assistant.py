import hashlib
import json
import os
from datetime import date, timedelta

import requests
import streamlit as st

from backend.config import load_project_environment
from backend.runtime import ensure_backend_available
from backend.university_sources import UNIVERSITY_DATA_DIR


load_project_environment()

st.set_page_config(
    page_title="Bundelkhand University AI Student Assistant",
    page_icon="🎓",
    layout="wide",
)
st.title("🎓 Bundelkhand University Student Assistant")
st.caption(
    "Jhansi · Ask about courses, admissions, fees, exams, policies, hostel, "
    "notices and departments."
)

api_url = os.getenv(
    "CAMPUSGUIDE_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")
api_key = os.getenv("CAMPUSGUIDE_API_KEY")
if not api_key:
    st.error(
        "API access is not configured. Set CAMPUSGUIDE_API_KEY in the local "
        "environment and restart the app."
    )
    st.stop()

try:
    api_url = ensure_backend_available(api_url)
except RuntimeError as error:
    st.error(f"The university assistant backend could not start: {error}")
    st.stop()

headers = {"X-CampusGuide-Key": api_key}
BTECH_CSE_SYLLABUS_URL = (
    "https://bujhansi.ac.in/media/3wophicd/"
    "b-tech-computer-science-and-engg.pdf"
)


def show_btech_cse_syllabus_download(
    answer: str,
    key: str,
) -> None:
    if BTECH_CSE_SYLLABUS_URL not in answer:
        return
    pdf_path = UNIVERSITY_DATA_DIR / "btech_cse_syllabus.pdf"
    if not pdf_path.is_file():
        st.error(
            "The official B.Tech CSE PDF is not cached locally. Use the "
            "official PDF link above or refresh this source."
        )
        return
    st.download_button(
        "Download official BU B.Tech CSE syllabus PDF",
        data=pdf_path.read_bytes(),
        file_name="BU_BTech_CSE_syllabus_effective_2017-18.pdf",
        mime="application/pdf",
        key=key,
    )


def show_http_error(action: str, error: requests.HTTPError) -> None:
    response = error.response
    if response is None:
        st.error(f"{action} failed: {error}")
        return
    try:
        detail = response.json().get("detail", response.text[:500])
    except ValueError:
        detail = response.text[:500]
    st.error(f"{action} failed (HTTP {response.status_code}): {detail}")


with st.sidebar:
    st.header("Student context")
    department = st.text_input(
        "Your course / department (optional)",
        placeholder="e.g. BCA, MBA, MCA, B.Tech",
        max_chars=160,
        help="You can enter any Bundelkhand University course or department.",
    )
    st.caption(
        "Your context is used to focus answers; it is not stored as a profile."
    )
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption(
        "Answers use the currently indexed official BU sources. A source "
        "gap is reported instead of guessed."
    )

st.subheader("Official university knowledge base")
try:
    sources_response = requests.get(
        f"{api_url}/api/university/sources",
        headers=headers,
        timeout=(3, 10),
    )
    sources_response.raise_for_status()
    source_data = sources_response.json()
except requests.HTTPError as error:
    show_http_error("Loading the university source list", error)
    source_data = None
except requests.RequestException as error:
    st.error(f"Could not load source status: {error}")
    source_data = None

if source_data is not None:
    available_sources = [
        source for source in source_data["sources"] if source["available"]
    ]
    st.info(
        f"{source_data['available_count']} of {len(source_data['sources'])} "
        "official BU source files are available in the local knowledge base. "
        "Department-wise syllabi and exam PDFs are discovered from BU's "
        "official live indexes; other policies/notices may not be exhaustive."
    )
    with st.expander("View indexed and pending official sources"):
        for source in source_data["sources"]:
            status_label = (
                "Downloaded; integrity checked when search index loads"
                if source["available"]
                else "Not downloaded"
            )
            st.markdown(
                f"- [{source['title']}]({source['url']}) — "
                f"**{status_label}** · {source['academic_year']}"
            )
            st.caption(f"Coverage: {source['coverage']}")
            if source.get("last_modified"):
                st.caption(
                    f"Official source last modified: "
                    f"{source['last_modified']}"
                )
            elif source.get("last_fetched"):
                st.caption(f"Last checked: {source['last_fetched']}")
        if st.button("Reload assistant search index"):
            try:
                reload_response = requests.post(
                    f"{api_url}/api/university/reload",
                    headers=headers,
                    timeout=(3, 10),
                )
                reload_response.raise_for_status()
            except requests.HTTPError as error:
                show_http_error("Reloading the BU search index", error)
            except requests.RequestException as error:
                st.error(f"Could not reload the BU search index: {error}")
            else:
                st.success(
                    "Search index cleared. It will rebuild from the "
                    "downloaded official sources on the next question."
                )
    if not available_sources:
        st.warning(
            "No official university files are indexed yet. From the project "
            "folder, run `python -m backend.refresh_university_sources`."
        )

st.divider()
st.markdown(
    """
    **Try asking:** “What are the 2026–27 BCA admission requirements?” ·
    “MBA ka latest syllabus kya hai?” · “Hostel charges ka official memo
    kya kehta hai?” · “B.Tech exam notice ka link do.”

    Fees, deadlines and regulations can change. Check the cited official
    university source and its applicable academic session before acting.
    """
)

with st.expander("Ask a question about your own PDF or plan syllabus study"):
    st.caption(
        "Upload a text-based PDF, up to 15 MB and 100 pages. The document is "
        "processed temporarily and sent to the configured AI provider; do not "
        "upload confidential records. Scanned PDFs need OCR first."
    )
    uploaded_pdf = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        max_upload_size=15,
        key="university_document_uploader",
    )
    if uploaded_pdf is not None:
        pdf_bytes = uploaded_pdf.getvalue()
        pdf_digest = hashlib.sha256(pdf_bytes).hexdigest()
        if st.session_state.get("document_pdf_digest") != pdf_digest:
            st.session_state.document_pdf_digest = pdf_digest
            st.session_state.document_pdf_history = []
            st.session_state.study_plan_result = None

        def submit_document_analysis(
            mode: str,
            question: str = "",
        ) -> None:
            try:
                response = requests.post(
                    f"{api_url}/api/documents/analyze",
                    headers=headers,
                    files={
                        "file": (
                            uploaded_pdf.name,
                            pdf_bytes,
                            "application/pdf",
                        )
                    },
                    data={"mode": mode, "question": question},
                    timeout=(10, 300),
                )
                response.raise_for_status()
                result = response.json()
            except requests.HTTPError as error:
                show_http_error("PDF analysis", error)
            except requests.RequestException as error:
                st.error(f"Could not reach the document service: {error}")
            else:
                st.session_state.document_pdf_history.append(
                    {
                        "mode": mode,
                        "question": question,
                        "answer": result["answer"],
                        "citations": result["citations"],
                    }
                )

        st.caption(f"Selected PDF: {uploaded_pdf.name}")
        if st.button("Summarize this PDF"):
            with st.spinner("Summarizing the uploaded PDF..."):
                submit_document_analysis("summary")

        with st.form("university_pdf_question_form"):
            pdf_question = st.text_input(
                "Question about your PDF",
                max_chars=2000,
            )
            ask_pdf = st.form_submit_button("Ask this PDF")
        if ask_pdf:
            if not pdf_question.strip():
                st.warning("Enter a question about the uploaded PDF.")
            else:
                with st.spinner("Finding cited passages..."):
                    submit_document_analysis(
                        "question",
                        pdf_question.strip(),
                    )

        st.markdown("#### Syllabus study planner")
        with st.form("university_study_plan_form"):
            exam_date = st.date_input(
                "Exam date",
                value=date.today() + timedelta(days=14),
                min_value=date.today(),
                max_value=date.today() + timedelta(days=364),
            )
            daily_hours = st.slider(
                "Focused study hours per day",
                min_value=1.0,
                max_value=8.0,
                value=2.0,
                step=0.5,
            )
            create_plan = st.form_submit_button("Create syllabus study plan")

        if create_plan:
            try:
                with st.spinner("Building a cited study schedule..."):
                    response = requests.post(
                        f"{api_url}/api/documents/study-plan",
                        headers=headers,
                        files={
                            "file": (
                                uploaded_pdf.name,
                                pdf_bytes,
                                "application/pdf",
                            )
                        },
                        data={
                            "exam_date": exam_date.isoformat(),
                            "daily_hours": daily_hours,
                        },
                        timeout=(10, 300),
                    )
                    response.raise_for_status()
                    st.session_state.study_plan_result = response.json()
            except requests.HTTPError as error:
                show_http_error("Study plan generation", error)
            except requests.RequestException as error:
                st.error(f"Could not reach the study planner: {error}")

        plan = st.session_state.get("study_plan_result")
        if plan is not None:
            plan_id = hashlib.sha256(
                json.dumps(plan, sort_keys=True).encode("utf-8")
            ).hexdigest()[:12]
            st.markdown(f"##### {plan['title']}")
            st.write(plan["overview"])
            completed_days = 0
            for day_index, study_day in enumerate(plan["days"]):
                with st.expander(
                    f"{study_day['date']} · {study_day['focus']} · "
                    f"{study_day['planned_minutes']} min",
                    expanded=day_index == 0,
                ):
                    if st.checkbox(
                        "Mark this study day complete",
                        key=f"plan-{pdf_digest[:12]}-{plan_id}-{day_index}",
                    ):
                        completed_days += 1
                    for activity in study_day["activities"]:
                        st.markdown(
                            f"- **{activity['category'].title()} · "
                            f"{activity['topic']}** "
                            f"({activity['minutes']} min): {activity['task']}"
                        )
                        if activity["citations"]:
                            st.caption(
                                "Syllabus pages: "
                                + ", ".join(activity["citations"])
                            )
            if plan["days"]:
                st.progress(
                    completed_days / len(plan["days"]),
                    text=(
                        f"{completed_days} of {len(plan['days'])} "
                        "study days complete"
                    ),
                )
            st.caption(
                "Study suggestions and completion checkmarks are not official "
                "course guidance; progress lasts for this browser session."
            )

        for item in st.session_state.get("document_pdf_history", []):
            if item["mode"] == "question":
                st.markdown(f"**Question:** {item['question']}")
            else:
                st.markdown("**PDF summary**")
            st.markdown(item["answer"])
            if item["citations"]:
                st.caption("Sources: " + ", ".join(item["citations"]))

if "messages" not in st.session_state:
    st.session_state.messages = []

for message_index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        show_btech_cse_syllabus_download(
            message["content"],
            key=f"saved-btech-cse-syllabus-{message_index}",
        )
        if message.get("agent_actions"):
            with st.expander("Official source searches"):
                for action in message["agent_actions"]:
                    st.write(f"- {action}")

user_query = st.chat_input(
    "Ask anything about Bundelkhand University..."
)
if user_query:
    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner(
            "Searching official BU sources... The first search after an "
            "update can take several minutes to build the index."
        ):
            conversation = [
                {
                    "role": message["role"],
                    "content": message["content"],
                }
                for message in st.session_state.messages[-30:]
            ]
            try:
                response = requests.post(
                    f"{api_url}/api/chat",
                    headers=headers,
                    json={
                        "department": department.strip(),
                        "messages": conversation,
                    },
                    timeout=(5, 600),
                )
                response.raise_for_status()
                result = response.json()
            except requests.HTTPError as error:
                show_http_error("University question", error)
            except requests.RequestException as error:
                st.error(
                    "Could not reach the BU assistant service. "
                    f"Details: {error}"
                )
            else:
                st.markdown(result["answer"])
                show_btech_cse_syllabus_download(
                    result["answer"],
                    key=(
                        "current-btech-cse-syllabus-"
                        f"{hashlib.sha256(user_query.encode()).hexdigest()[:12]}"
                    ),
                )
                if result["agent_actions"]:
                    with st.expander("Official source searches"):
                        for action in result["agent_actions"]:
                            st.write(f"- {action}")
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "agent_actions": result["agent_actions"],
                    }
                )
