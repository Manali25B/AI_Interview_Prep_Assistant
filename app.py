"""Streamlit entry point for the AI Interview Preparation Assistant."""

import os
from statistics import fmean

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings

from assistant.config import build_chat_model, get_api_key, get_model_name
from assistant.core import source_references
from assistant.documents import create_vector_store, extract_uploaded_files, format_context, retrieve
from assistant.service import (
    evaluate_answer,
    extract_profile,
    generate_questions,
    summarize_mock_interview,
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NAV_ITEMS = ["Overview", "Resume analysis", "Question studio", "Answer evaluation", "Mock interview"]

st.set_page_config(
    page_title="Interview Prep Assistant",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def load_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def initialize_state() -> None:
    defaults = {
        "upload_generation": 0,
        "pages": [],
        "vector_store": None,
        "chunk_count": 0,
        "questions": [],
        "question_sources": [],
        "answer_result": None,
        "answer_sources": [],
        "evaluation_question_source": "Enter my own question",
        "evaluation_question": "",
        "evaluation_answer": "",
        "evaluation_in_progress": False,
        "_evaluation_previous_source": None,
        "_evaluation_previous_generated_index": None,
        "mock_questions": [],
        "mock_results": [],
        "mock_summary": None,
        "mock_complete": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def current_generation() -> int:
    return int(st.session_state.upload_generation)


def current_job_description() -> str:
    return st.session_state.get(f"job_description_{current_generation()}", "")


def navigate_to(page: str) -> None:
    st.session_state.active_page = page
    st.session_state[f"page_navigation_{current_generation()}"] = page


def sync_active_page(widget_key: str) -> None:
    st.session_state.active_page = st.session_state[widget_key]


def indexed() -> bool:
    return st.session_state.vector_store is not None


def load_model_or_report() -> object | None:
    api_key = get_api_key()
    if not api_key:
        st.error("OpenRouter is not configured. Add OPENROUTER_API_KEY to `.env` or Streamlit Secrets.")
        return None
    try:
        return build_chat_model(api_key, get_model_name())
    except ValueError as exc:
        st.error(str(exc))
        return None


def show_model_error(error: Exception) -> None:
    if isinstance(error, ValueError):
        st.error(str(error))
        return
    status = getattr(error, "status_code", None)
    if status == 429:
        st.error("The free model is rate-limited right now. Wait a moment and retry.")
    elif status in (401, 403):
        st.error("OpenRouter rejected the configured key or model access. Check your secret and free-tier access.")
    else:
        error_type = type(error).__name__
        st.error(
            f"Free-model request failed ({error_type}). Check free-model availability and your connection, then retry. No paid fallback is used."
        )


def render_references(documents: list) -> None:
    references = source_references(documents)
    if references:
        st.caption("Sources: " + "  ·  ".join(references))


def render_header(eyebrow: str, title: str, description: str) -> None:
    st.markdown(
        f'<div class="page-heading"><div class="eyebrow">{eyebrow}</div>'
        f'<h1>{title}</h1><p>{description}</p></div>',
        unsafe_allow_html=True,
    )


def render_empty_state(title: str, body: str) -> None:
    st.markdown(
        f'<div class="empty-state"><div class="empty-mark">+</div>'
        f'<strong>{title}</strong><span>{body}</span></div>',
        unsafe_allow_html=True,
    )


def render_profile() -> None:
    render_header("CANDIDATE PROFILE", "Resume analysis", "Turn your resume into a grounded, searchable interview brief.")
    resume_pages = [page for page in st.session_state.pages if page.metadata.get("document_type") == "resume"]
    if not resume_pages:
        render_empty_state("No resume indexed", "Add a resume PDF in the document library to extract your skills, projects, and experience.")
        return

    left, right = st.columns([3, 1])
    with left:
        st.markdown("#### Resume on file")
        st.write(f"{resume_pages[0].metadata.get('source', 'Resume')} · {len(resume_pages)} text pages")
    with right:
        st.markdown(f'<div class="metric-box"><span>Indexed chunks</span><strong>{st.session_state.chunk_count}</strong></div>', unsafe_allow_html=True)

    if st.button("Analyze resume", type="primary", icon=":material/analytics:"):
        model = load_model_or_report()
        if model:
            with st.spinner("Reviewing resume evidence…"):
                try:
                    context = format_context(resume_pages, max_chars=14000)
                    st.session_state.resume_profile = extract_profile(model, context)
                    st.session_state.resume_sources = resume_pages
                except Exception as error:
                    show_model_error(error)
    profile = st.session_state.get("resume_profile")
    if profile:
        st.divider()
        columns = st.columns(3)
        for column, label, key in zip(columns, ("Skills", "Projects", "Experience"), ("skills", "projects", "experience")):
            with column:
                st.markdown(f"#### {label}")
                values = profile.get(key, [])
                if values:
                    for value in values:
                        st.markdown(f"- {value}")
                else:
                    st.caption("No explicit evidence found in the resume.")
        render_references(resume_pages)


def render_question_studio() -> None:
    render_header("PRACTICE DESIGN", "Question studio", "Build a focused question set from the target role and your uploaded material.")
    description = st.text_area(
        "Target job description",
        key=f"job_description_{current_generation()}",
        height=145,
        placeholder="Paste the role description to focus questions on its responsibilities and requirements…",
    )
    count = st.slider("Number of questions", min_value=4, max_value=12, value=8)
    if st.button("Generate question set", type="primary", icon=":material/auto_awesome:"):
        model = load_model_or_report()
        if model:
            documents = retrieve(st.session_state.vector_store, description or "target role interview questions", count=6) if indexed() else []
            with st.spinner("Building your question set…"):
                try:
                    context = format_context(documents)
                    st.session_state.questions = generate_questions(
                        model, job_description=description, context=context, count=count
                    )
                    st.session_state.question_sources = documents
                except Exception as error:
                    show_model_error(error)

    questions = st.session_state.questions
    if questions:
        st.divider()
        category = st.selectbox("Question category", ["All categories", "Technical", "HR", "Project-based", "Scenario-based"])
        filtered = questions if category == "All categories" else [item for item in questions if item["category"] == category]
        for index, item in enumerate(filtered, start=1):
            with st.container(border=True):
                st.caption(f"{item['category'].upper()}  ·  {index:02d}")
                st.markdown(f"**{item['question']}**")
                if item.get("why"):
                    st.caption(item["why"])
        render_references(st.session_state.question_sources)


def render_answer_result(result: dict, documents: list) -> None:
    score_col, feedback_col = st.columns([1, 3])
    with score_col:
        st.markdown(f'<div class="score-box"><span>ANSWER SCORE</span><strong>{result["score"]:g}<small> / 10</small></strong></div>', unsafe_allow_html=True)
    with feedback_col:
        st.markdown("#### Coaching notes")
        st.write(result["feedback"])
    strengths = result.get("strengths", [])
    if strengths:
        st.markdown("#### Strengths")
        for strength in strengths:
            st.markdown(f"- {strength}")
    if result["missing_points"]:
        st.markdown("#### Points to strengthen")
        for point in result["missing_points"]:
            st.markdown(f"- {point}")
    st.markdown("#### A stronger version")
    with st.container(border=True):
        st.text(result["improved_answer"])
    render_references(documents)


def render_answer_evaluation() -> None:
    render_header("ANSWER WORKSHOP", "Answer evaluation", "Get a score, specific coaching, and a more effective version of your answer.")
    questions = st.session_state.questions
    source = st.radio(
        "Question source",
        options=["Select a generated question", "Enter my own question"],
        horizontal=True,
        key="evaluation_question_source",
    )

    previous_source = st.session_state._evaluation_previous_source
    previous_index = st.session_state._evaluation_previous_generated_index
    selected_index = None
    if source == "Select a generated question":
        if not questions:
            if previous_source != source:
                st.session_state.evaluation_question = ""
            st.info("No generated questions yet. Create questions in Question Studio, or choose ‘Enter my own question’.")
        else:
            selected_index = st.selectbox(
                "Generated questions",
                options=list(range(len(questions))),
                format_func=lambda index: f"{questions[index]['category']} · {questions[index]['question']}",
                key="evaluation_generated_question",
            )
            if previous_source != source or previous_index != selected_index:
                st.session_state.evaluation_question = questions[selected_index]["question"]
    elif previous_source != source:
        st.session_state.evaluation_question = ""

    st.session_state._evaluation_previous_source = source
    st.session_state._evaluation_previous_generated_index = selected_index

    if source == "Enter my own question":
        question = st.text_input(
            "Interview question",
            key="evaluation_question",
            max_chars=1000,
            placeholder="Type the question you want to practice…",
        )
    elif questions:
        question = st.text_input(
            "Interview question",
            key="evaluation_question",
            disabled=True,
        )
    else:
        question = ""

    answer = st.text_area(
        "Your answer",
        height=210,
        placeholder="Write the answer you would give in an interview…",
        key="evaluation_answer",
    )
    valid_inputs = bool(question.strip()) and bool(answer.strip())
    if not valid_inputs:
        st.caption("Enter both an interview question and your answer to continue.")
    if st.button(
        "Evaluate answer",
        type="primary",
        icon=":material/rate_review:",
        disabled=not valid_inputs or st.session_state.evaluation_in_progress,
    ):
        st.session_state.evaluation_in_progress = True
        try:
            model = load_model_or_report()
            if model:
                documents = retrieve(st.session_state.vector_store, question + "\n" + answer, count=5) if indexed() else []
                with st.spinner("Evaluating your response…"):
                    result = evaluate_answer(
                        model,
                        question=question.strip(),
                        answer=answer.strip(),
                        job_description=current_job_description(),
                        context=format_context(documents),
                    )
                    st.session_state.answer_result = result
                    st.session_state.answer_sources = documents
                    st.session_state.answer_result_question = question.strip()
                    st.session_state.answer_result_answer = answer.strip()
        except Exception as error:
            show_model_error(error)
        finally:
            st.session_state.evaluation_in_progress = False

    result_matches_inputs = (
        st.session_state.answer_result
        and st.session_state.get("answer_result_question") == question.strip()
        and st.session_state.get("answer_result_answer") == answer.strip()
    )
    if result_matches_inputs:
        st.divider()
        render_answer_result(st.session_state.answer_result, st.session_state.answer_sources)


def render_mock_result(record: dict, index: int) -> None:
    result = record["evaluation"]
    with st.expander(f"Question {index + 1} · {result['score']:g}/10", expanded=False):
        st.markdown(f"**{record['question']}**")
        st.caption("Your answer")
        st.write(record["answer"])
        st.caption(result["feedback"])
        render_references(record["sources"])


def render_mock_interview() -> None:
    render_header("LIVE PRACTICE", "Mock interview", "Work through one prompt at a time, then review your overall performance.")
    if not st.session_state.mock_questions and not st.session_state.mock_complete:
        st.markdown("#### Set up your practice round")
        st.caption("Five questions across technical, behavioral, project, and scenario-based topics.")
        if st.button("Start mock interview", type="primary", icon=":material/play_arrow:"):
            model = load_model_or_report()
            if model:
                documents = retrieve(st.session_state.vector_store, current_job_description() or "interview role", count=6) if indexed() else []
                with st.spinner("Preparing your interview…"):
                    try:
                        st.session_state.mock_questions = generate_questions(
                            model,
                            job_description=current_job_description(),
                            context=format_context(documents),
                            count=5,
                        )
                        st.session_state.mock_results = []
                        st.session_state.mock_summary = None
                    except Exception as error:
                        show_model_error(error)
        return

    if st.session_state.mock_complete:
        results = st.session_state.mock_results
        average = fmean(item["evaluation"]["score"] for item in results) if results else 0
        st.markdown(
            f'<div class="completion-banner"><span>ROUND COMPLETE</span><strong>{average:.1f}<small> / 10 average</small></strong></div>',
            unsafe_allow_html=True,
        )
        summary = st.session_state.mock_summary
        if summary:
            st.markdown("#### Overall readout")
            st.write(summary["summary"])
            left, right = st.columns(2)
            with left:
                st.markdown("**Strengths**")
                for item in summary["strengths"]:
                    st.markdown(f"- {item}")
            with right:
                st.markdown("**Practice priorities**")
                for item in summary["priorities"]:
                    st.markdown(f"- {item}")
        for index, record in enumerate(results):
            render_mock_result(record, index)
        if st.button("Start a new round", icon=":material/replay:"):
            st.session_state.mock_questions = []
            st.session_state.mock_results = []
            st.session_state.mock_summary = None
            st.session_state.mock_complete = False
            st.rerun()
        return

    questions = st.session_state.mock_questions
    completed = len(st.session_state.mock_results)
    total = len(questions)
    st.progress(completed / total, text=f"Question {min(completed + 1, total)} of {total}")
    for index, record in enumerate(st.session_state.mock_results):
        render_mock_result(record, index)

    question = questions[completed]
    with st.container(border=True):
        st.caption(f"{question['category'].upper()}  ·  QUESTION {completed + 1:02d}")
        st.markdown(f"### {question['question']}")
        answer = st.text_area("Your response", height=200, key=f"mock_answer_{completed}", placeholder="Answer as though you were speaking to the interviewer…")
        if st.button("Submit answer", type="primary", icon=":material/arrow_forward:", disabled=not answer.strip()):
            model = load_model_or_report()
            if model:
                documents = retrieve(st.session_state.vector_store, question["question"] + "\n" + answer, count=5) if indexed() else []
                with st.spinner("Reviewing your answer…"):
                    try:
                        evaluation = evaluate_answer(
                            model,
                            question=question["question"],
                            answer=answer,
                            job_description=current_job_description(),
                            context=format_context(documents),
                        )
                        st.session_state.mock_results.append({
                            "question": question["question"],
                            "answer": answer,
                            "evaluation": evaluation,
                            "sources": documents,
                        })
                        if len(st.session_state.mock_results) == total:
                            st.session_state.mock_complete = True
                            try:
                                st.session_state.mock_summary = summarize_mock_interview(model, st.session_state.mock_results)
                            except Exception:
                                st.session_state.mock_summary = {
                                    "summary": "Your practice round is complete. Review the feedback on each response to choose your next focus.",
                                    "strengths": [],
                                    "priorities": [],
                                }
                        st.rerun()
                    except Exception as error:
                        show_model_error(error)


def render_overview() -> None:
    render_header("INTERVIEW PREPARATION", "Make every answer count.", "A private, evidence-grounded workspace for your next interview.")
    pages = st.session_state.pages
    documents = len({page.metadata.get("source") for page in pages})
    metrics = st.columns(4)
    metric_data = [
        ("Documents", documents, "PDFs in this session"),
        ("Indexed chunks", st.session_state.chunk_count, "Available for retrieval"),
        ("Questions", len(st.session_state.questions), "In your latest set"),
        ("Mock rounds", int(st.session_state.mock_complete), "Completed this session"),
    ]
    for column, (label, value, detail) in zip(metrics, metric_data):
        with column:
            st.markdown(f'<div class="metric-box"><span>{label}</span><strong>{value}</strong><small>{detail}</small></div>', unsafe_allow_html=True)

    st.markdown("## Your preparation desk")
    first, second = st.columns(2)
    with first:
        with st.container(border=True):
            st.markdown("#### 01 · Know your story")
            st.write("Extract resume evidence into a clear view of your skills, projects, and experience.")
            st.button("Open resume analysis", key="go_resume", icon=":material/arrow_forward:", on_click=navigate_to, args=("Resume analysis",))
        with st.container(border=True):
            st.markdown("#### 02 · Rehearse under pressure")
            st.write("Practice a structured interview, one response at a time, with feedback after each answer.")
            st.button("Open mock interview", key="go_mock", icon=":material/arrow_forward:", on_click=navigate_to, args=("Mock interview",))
    with second:
        with st.container(border=True):
            st.markdown("#### 03 · Prepare for the role")
            st.write("Generate technical, HR, project, and scenario questions from your target role.")
            st.button("Open question studio", key="go_questions", icon=":material/arrow_forward:", on_click=navigate_to, args=("Question studio",))
        with st.container(border=True):
            st.markdown("#### 04 · Improve your delivery")
            st.write("Get a 10-point score, missing points, and an improved version grounded in your answer.")
            st.button("Evaluate an answer", key="go_evaluation", icon=":material/arrow_forward:", on_click=navigate_to, args=("Answer evaluation",))

    if not indexed():
        st.info("Add a resume or study PDF in the sidebar document library, then index it to enable source-grounded features.", icon=":material/info:")
    if not get_api_key():
        st.warning("Add your OpenRouter API key in `.env` for local development or Streamlit Secrets when deployed.", icon=":material/key:")


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
        :root { --ink: #e8eef2; --muted: #91a2ac; --line: #26343d; --panel: #131c24; --mint: #48d6b5; --amber: #edb96b; }
        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
        h1, h2, h3, h4 { font-family: 'Manrope', sans-serif; letter-spacing: 0 !important; }
        .stApp { background: radial-gradient(ellipse at 88% 0%, rgba(72,214,181,.08), transparent 32%), #0c1117; }
        [data-testid="stSidebar"] { background: #10171e; border-right: 1px solid var(--line); }
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .55rem; }
        .brand { padding: .4rem 0 1.1rem; border-bottom: 1px solid var(--line); margin-bottom: .8rem; }
        .brand-mark { color: var(--mint); font: 800 1.1rem Manrope,sans-serif; }
        .brand-name { color: var(--ink); font: 700 1rem Manrope,sans-serif; margin-top: .2rem; }
        .brand-caption { color: var(--muted); font-size: .75rem; margin-top: .2rem; }
        .page-heading { padding: .2rem 0 1rem; }
        .eyebrow { color: var(--mint); font: 700 .69rem 'DM Sans',sans-serif; letter-spacing: .11em; }
        .page-heading h1 { color: var(--ink); font-size: clamp(1.8rem, 3vw, 2.55rem); margin: .25rem 0 .35rem; }
        .page-heading p { color: var(--muted); max-width: 680px; margin: 0; }
        .metric-box, .score-box { background: linear-gradient(145deg, #17222a, #121a21); border: 1px solid var(--line); border-radius: 8px; padding: 1rem 1.05rem; min-height: 112px; }
        .metric-box span, .score-box span { color: var(--muted); display: block; font-size: .78rem; }
        .metric-box strong { color: var(--ink); display: block; font: 700 1.7rem Manrope,sans-serif; margin: .35rem 0 .12rem; }
        .metric-box small { color: #81919a; font-size: .72rem; }
        .score-box { min-height: 130px; display: flex; flex-direction: column; justify-content: center; }
        .score-box strong, .completion-banner strong { color: var(--mint); font: 800 2.2rem Manrope,sans-serif; }
        .score-box small, .completion-banner small { color: var(--muted); font: 500 .85rem 'DM Sans',sans-serif; }
        .empty-state { margin: 1rem 0; border: 1px dashed #34454f; border-radius: 8px; padding: 2rem; display:flex; flex-direction:column; gap:.45rem; color:var(--muted); }
        .empty-state strong { color:var(--ink); font:600 1rem Manrope,sans-serif; }
        .empty-mark { color:var(--mint); font-size:1.2rem; }
        .answer-sample { color: var(--ink); background:#121b22; border-left:3px solid var(--mint); padding:1rem 1.15rem; line-height:1.65; white-space:pre-wrap; border-radius:0 6px 6px 0; }
        .completion-banner { background:linear-gradient(100deg,rgba(72,214,181,.12),rgba(237,185,107,.07)); border:1px solid #315047; border-radius:8px; padding:1.2rem 1.4rem; display:flex; align-items:center; justify-content:space-between; margin:.5rem 0 1rem; }
        .completion-banner span { color:var(--mint); font-size:.75rem; font-weight:700; letter-spacing:.08em; }
        div[data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--line); border-radius: 8px; }
        .stButton button[kind="primary"] { background:var(--mint); color:#09211b; border-color:var(--mint); font-weight:700; }
        .stButton button { border-radius:6px; }
        div[data-baseweb="select"] > div, .stTextInput input, .stTextArea textarea { border-radius:6px; }
        [data-testid="stProgressBar"] > div > div { background-color:var(--mint); }
        hr { border-color:var(--line); }
        @media (max-width: 720px) { .page-heading { padding-top:.6rem; } .completion-banner { align-items:flex-start; flex-direction:column; gap:.5rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    generation = current_generation()
    with st.sidebar:
        st.markdown(
            '<div class="brand"><div class="brand-mark">✦</div><div class="brand-name">Interview Prep</div>'
            '<div class="brand-caption">Your private practice desk</div></div>',
            unsafe_allow_html=True,
        )
        navigation_key = f"page_navigation_{generation}"
        selected = st.radio(
            "Workspace",
            NAV_ITEMS,
            index=NAV_ITEMS.index(st.session_state.get("active_page", "Overview")),
            label_visibility="collapsed",
            key=navigation_key,
            on_change=sync_active_page,
            args=(navigation_key,),
        )
        st.session_state.active_page = selected
        st.markdown("##### DOCUMENT LIBRARY")
        resume = st.file_uploader("Resume PDF", type=["pdf"], key=f"resume_upload_{generation}")
        study_files = st.file_uploader(
            "Study materials · PDFs", type=["pdf"], accept_multiple_files=True, key=f"study_upload_{generation}"
        )
        if st.button("Index selected PDFs", use_container_width=True, disabled=not resume and not study_files):
            st.session_state.vector_store = None
            st.session_state.pages = []
            st.session_state.chunk_count = 0
            for key, value in (
                ("resume_profile", None), ("resume_sources", []), ("questions", []),
                ("question_sources", []), ("answer_result", None), ("answer_sources", []),
                ("mock_questions", []), ("mock_results", []), ("mock_summary", None),
                ("mock_complete", False),
            ):
                st.session_state[key] = value
            try:
                pages = extract_uploaded_files(resume, study_files or [])
            except ValueError as error:
                st.error(str(error))
            else:
                with st.spinner("Embedding documents on this device…"):
                    try:
                        vector_store, chunk_count = create_vector_store(pages, load_embeddings())
                        st.session_state.pages = pages
                        st.session_state.vector_store = vector_store
                        st.session_state.chunk_count = chunk_count
                        st.session_state.resume_profile = None
                        st.session_state.questions = []
                        st.session_state.answer_result = None
                        st.session_state.mock_questions = []
                        st.session_state.mock_results = []
                        st.session_state.mock_summary = None
                        st.session_state.mock_complete = False
                        st.success(f"Indexed {len(pages)} pages into {chunk_count} searchable chunks.")
                    except Exception:
                        st.error("Document indexing failed. Check the PDF text and available memory, then retry.")
        if indexed():
            document_count = len({page.metadata.get("source") for page in st.session_state.pages})
            st.caption(f"{document_count} PDFs · {st.session_state.chunk_count} chunks in this session")
        st.divider()
        st.caption("MODEL · OPENROUTER FREE")
        try:
            st.caption(get_model_name())
        except ValueError as error:
            st.error(str(error))
        st.caption("API key configured" if get_api_key() else "API key not configured")
        st.caption("Document excerpts and answers are sent to OpenRouter only when you request an AI action.")
        if st.button("Clear session data", use_container_width=True, icon=":material/delete_outline:"):
            for key in (
                "pages", "vector_store", "chunk_count", "questions", "question_sources",
                "answer_result", "answer_sources", "resume_profile", "resume_sources",
                "mock_questions", "mock_results", "mock_summary", "mock_complete",
            ):
                st.session_state.pop(key, None)
            st.session_state.upload_generation = generation + 1
            st.session_state.active_page = "Overview"
            st.rerun()
    return selected


def main() -> None:
    initialize_state()
    apply_styles()
    page = render_sidebar()
    if page == "Overview":
        render_overview()
    elif page == "Resume analysis":
        render_profile()
    elif page == "Question studio":
        render_question_studio()
    elif page == "Answer evaluation":
        render_answer_evaluation()
    elif page == "Mock interview":
        render_mock_interview()


if __name__ == "__main__":
    main()