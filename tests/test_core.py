from io import BytesIO

import pytest
from pypdf import PdfWriter

from assistant.config import get_model_name, is_free_model
from assistant.core import parse_json_object, source_references
from assistant.documents import (
    MAX_DOCUMENT_COUNT,
    MAX_UPLOAD_BYTES,
    extract_pdf_pages,
    extract_uploaded_files,
)
from assistant.service import evaluate_answer, generate_questions


def test_accepts_only_explicit_free_model_ids():
    assert is_free_model("openrouter/free")
    assert is_free_model("meta-llama/example:free")
    assert not is_free_model("provider/model")
    assert not is_free_model("provider/model:free-ish")


def test_rejects_paid_model_configuration(monkeypatch):
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    with pytest.raises(ValueError, match="ending in ':free'"):
        get_model_name({"OPENROUTER_MODEL": "provider/paid-model"})


def test_default_model_is_free_router(monkeypatch):
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    assert get_model_name({}) == "openrouter/free"


def test_parses_json_inside_markdown_fence():
    assert parse_json_object('```json\n{"score": 8}\n```') == {"score": 8}


def test_rejects_invalid_json():
    with pytest.raises(ValueError, match="invalid JSON"):
        parse_json_object("{not json}")


def test_source_references_are_deduplicated_by_page():
    class StubDocument:
        def __init__(self, source, page):
            self.metadata = {"source": source, "page": page}

    documents = [StubDocument("resume.pdf", 1), StubDocument("resume.pdf", 1), StubDocument("notes.pdf", 2)]
    assert source_references(documents) == ["resume.pdf · p. 1", "notes.pdf · p. 2"]


def test_answer_evaluation_uses_mocked_model_and_validates_score():
    class StubModel:
        def invoke(self, messages):
            class Response:
                content = '{"score": 8.5, "strengths": ["Clear and relevant."], "feedback": "Use a concrete example.", "missing_points": ["Add an example."], "improved_answer": "A clearer answer."}'

            return Response()

    result = evaluate_answer(
        StubModel(),
        question="How do you prioritize work?",
        answer="I sort by impact.",
        job_description="",
        context="",
    )
    assert result["score"] == 8.5
    assert result["strengths"] == ["Clear and relevant."]
    assert result["missing_points"] == ["Add an example."]


def test_answer_evaluation_rejects_out_of_range_score():
    class StubModel:
        def invoke(self, messages):
            class Response:
                content = '{"score": 11, "feedback": "Feedback", "missing_points": [], "improved_answer": "Answer"}'

            return Response()

    with pytest.raises(ValueError, match="outside the 0 to 10 range"):
        evaluate_answer(
            StubModel(), question="Question", answer="Answer", job_description="", context=""
        )


def test_question_generation_retries_non_json_response():
    class StubModel:
        def __init__(self, responses):
            self.responses = responses
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1

            class Response:
                pass

            response = Response()
            response.content = self.responses.pop(0)
            return response

    model = StubModel([
        "Here are some questions for the role.",
        '{"questions":['
        '{"category":"Technical","question":"How would you design an API?","why":"Role relevance"},'
        '{"category":"HR","question":"What motivates you?","why":"Communication"},'
        '{"category":"Project-based","question":"Describe a project.","why":"Evidence"},'
        '{"category":"Scenario-based","question":"How would you handle an outage?","why":"Judgment"}]}',
    ])

    questions = generate_questions(
        model, job_description="Backend engineer", context="", count=4
    )

    assert model.calls == 2
    assert len(questions) == 4
    assert {item["category"] for item in questions} == {
        "Technical", "HR", "Project-based", "Scenario-based"
    }


def test_json_retry_fails_with_clear_message_after_second_malformed_response():
    class StubModel:
        def invoke(self, messages):
            class Response:
                content = "Still not JSON."

            return Response()

    with pytest.raises(ValueError, match="unstructured response twice"):
        generate_questions(
            StubModel(), job_description="Backend engineer", context="", count=4
        )


def test_pdf_validation_rejects_empty_and_invalid_uploads():
    with pytest.raises(ValueError, match="is empty"):
        extract_pdf_pages(b"", "empty.pdf", "resume")
    with pytest.raises(ValueError, match="valid PDF"):
        extract_pdf_pages(b"not a PDF", "broken.pdf", "resume")


def test_pdf_validation_rejects_pages_without_extractable_text():
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    pdf_bytes = BytesIO()
    writer.write(pdf_bytes)
    with pytest.raises(ValueError, match="No selectable text"):
        extract_pdf_pages(pdf_bytes.getvalue(), "scanned.pdf", "resume")


def test_upload_limits_are_checked_before_reading_documents():
    class Upload:
        name = "large.pdf"
        size = MAX_UPLOAD_BYTES + 1

        def getvalue(self):
            raise AssertionError("Oversized upload should not be read")

    with pytest.raises(ValueError, match="30 MB"):
        extract_uploaded_files(None, [Upload()])
    with pytest.raises(ValueError, match=str(MAX_DOCUMENT_COUNT)):
        extract_uploaded_files(None, [Upload() for _ in range(MAX_DOCUMENT_COUNT + 1)])