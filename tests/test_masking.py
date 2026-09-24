"""M11: masking personal data for cloud models."""

import json

from querynest.masking import Masker, StreamUnmasker


def test_mask_tool_result_and_unmask_answer():
    m = Masker(columns=["customername"])
    content = json.dumps({"rows": [{"customername": "ACME LLC", "revenue": 10}, {"customername": "Beta Co", "revenue": 5}]})
    masked = json.loads(m.mask_tool_content(content))
    assert masked["rows"][0]["customername"] == "[P1]" and masked["rows"][0]["revenue"] == 10
    assert "ACME" not in json.dumps(masked)
    assert m.unmask_text("Top customer is [P1], then [P2].") == "Top customer is ACME LLC, then Beta Co."


def test_same_value_same_token_and_text_masking():
    m = Masker(columns=["customername"])
    m.mask_json({"customername": "ACME LLC"})
    assert m.mask_json({"customername": "ACME LLC"}) == {"customername": "[P1]"}
    assert m.mask_text("How much did ACME LLC buy?") == "How much did [P1] buy?"


def test_unmask_sql_escapes_quotes():
    m = Masker({"[P1]": "O'Brien LLC"})
    assert m.unmask_sql("SELECT 1 WHERE c = '[P1]'") == "SELECT 1 WHERE c = 'O''Brien LLC'"


def test_stream_unmasker_handles_split_tokens():
    s = StreamUnmasker(Masker({"[P12]": "ACME"}))
    out = s.feed("Best: [P") + s.feed("1") + s.feed("2] wins") + s.flush()
    assert out == "Best: ACME wins"


def test_mapping_persists_between_questions():
    first = Masker(columns=["customername"])
    first.mask_json({"customername": "ACME"})
    again = Masker(first.mapping, columns=["customername"])
    assert again.mask_json({"customername": "ACME"}) == {"customername": "[P1]"}
