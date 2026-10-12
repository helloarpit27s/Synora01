import pytest

from app.safety.guardrails import check_output_text, parse_json


@pytest.mark.parametrize(
    "text",
    [
        "You have gastritis, most likely.",
        "That sounds like an infection.",
        "You should take 20 mg twice a day.",
        "Start 500 mg of metformin with dinner.",
        "You're fine, nothing to worry about.",
        "You don't need to see a doctor for this.",
    ],
)
def test_unsafe_outputs_are_caught(text):
    assert check_output_text(text).safe is False


@pytest.mark.parametrize(
    "text",
    [
        "Nothing you logged crossed a referral threshold. Keep logging and come back if it sharpens.",
        "Burning after meals is a pattern worth raising with your clinician.",
        "Based on what you described, this needs emergency care now.",
        "Sodium adds up fastest in broth rather than at the table.",
    ],
)
def test_safe_outputs_pass(text):
    assert check_output_text(text).safe is True


def test_parse_json_handles_fences_and_garbage():
    assert parse_json('```json\n{"a": 1}\n```')["a"] == 1
    assert parse_json('here you go: {"b": 2} hope that helps')["b"] == 2
    assert parse_json("not json", {"fallback": True})["fallback"] is True
    assert parse_json("", {}) == {}
