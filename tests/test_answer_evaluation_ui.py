from pathlib import Path

from streamlit.testing.v1 import AppTest


def _open_evaluation_page():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path).run(timeout=30)
    workspace = next(widget for widget in app.radio if widget.label == "Workspace")
    workspace.set_value("Answer evaluation").run(timeout=30)
    return app


def _find_widget(widgets, label):
    return next(widget for widget in widgets if widget.label == label)


def test_manual_question_mode_requires_question_and_answer_then_enables_submit():
    app = _open_evaluation_page()
    source = _find_widget(app.radio, "Question source")
    assert source.options == ["Select a generated question", "Enter my own question"]
    assert source.value == "Enter my own question"
    assert _find_widget(app.button, "Evaluate answer").disabled

    _find_widget(app.text_input, "Interview question").set_value("How do you prioritize work?").run(timeout=30)
    _find_widget(app.text_area, "Your answer").set_value("I prioritize by impact and urgency.").run(timeout=30)

    assert not _find_widget(app.button, "Evaluate answer").disabled


def test_generated_question_populates_question_and_requires_answer():
    app = _open_evaluation_page()
    app.session_state["questions"] = [
        {"category": "Technical", "question": "How do you design a REST API?", "why": "Role fit"},
        {"category": "HR", "question": "What motivates you?", "why": "Communication"},
    ]
    app.run(timeout=30)
    _find_widget(app.radio, "Question source").set_value("Select a generated question").run(timeout=30)

    selector = _find_widget(app.selectbox, "Generated questions")
    question = _find_widget(app.text_input, "Interview question")
    assert selector.value == 0
    assert question.value == "How do you design a REST API?"
    assert question.disabled
    assert _find_widget(app.button, "Evaluate answer").disabled

    _find_widget(app.text_area, "Your answer").set_value("I define resources and clear contracts.").run(timeout=30)
    assert not _find_widget(app.button, "Evaluate answer").disabled


def test_generated_question_mode_without_questions_shows_guidance():
    app = _open_evaluation_page()
    _find_widget(app.radio, "Question source").set_value("Select a generated question").run(timeout=30)

    assert any("No generated questions yet" in item.value for item in app.info)
    assert not any(item.label == "Generated questions" for item in app.selectbox)
    assert _find_widget(app.button, "Evaluate answer").disabled