from cts.train.openmath_text import prompt_text_from_openmath_row


def test_openmath_question_field():
    row = {"question": " 2+2? ", "expected_answer": "4"}
    assert prompt_text_from_openmath_row(row) == "2+2?"


def test_openmath_fallback():
    row = {"problem": "x"}
    assert prompt_text_from_openmath_row(row) == "x"
