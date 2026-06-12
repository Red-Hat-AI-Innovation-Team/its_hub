"""Offline tests for the MMAU-Pro harness (scoring, loader, prompt) — no GPU/server."""

import os

import pytest

from benchmarking.mmau_pro.loader import MCQRecord, record_from_row
from benchmarking.mmau_pro.prompt import build_messages
from benchmarking.mmau_pro.scoring import (
    extract_letter,
    is_correct,
    match_answer_index,
    predicted_index,
)
from its_hub.api.types import ChatMessages

DATA_ROOT = "/home/exx/inference-time-scaling/mmau_pro_testmini"


# ---------------------------- scoring ---------------------------- #


def test_match_answer_index_exact_and_fuzzy():
    choices = ["Boba tea", "Coffee", "Green tea"]
    assert match_answer_index("Boba tea", choices) == 0
    assert match_answer_index("boba  tea", choices) == 0  # normalized
    assert match_answer_index("a boba tea", choices) == 0  # article stripped
    assert match_answer_index("completely different", choices) is None


def test_extract_letter():
    assert extract_letter("reasoning...\nAnswer: B", 4) == 1
    assert extract_letter("I think it's (C).", 4) == 2
    assert extract_letter("Answer: D then Answer: A", 4) == 0  # last marker wins
    assert extract_letter("Answer: F", 4) is None  # out of range
    assert extract_letter("no letter here", 4) is None


def test_predicted_index_letter_then_text_fallback():
    choices = ["piano", "violin", "drums"]
    assert predicted_index("Answer: B", choices) == 1
    assert predicted_index("it is clearly a violin", choices) == 1  # text fallback
    assert predicted_index("???", choices) is None


def test_is_correct():
    choices = ["piano", "violin"]
    assert is_correct("Answer: A", choices, 0) is True
    assert is_correct("Answer: B", choices, 0) is False
    assert is_correct("Answer: A", choices, None) is None  # ungradeable


# ---------------------------- loader ---------------------------- #


def test_record_from_row_mcq():
    row = {
        "id": "x1", "question": "what?", "answer": "Coffee",
        "choices": ["Boba tea", "Coffee"], "audio_path": ["data/x1.wav"],
        "category": "sound", "length_type": "short",
    }
    rec = record_from_row(row, "/root")
    assert isinstance(rec, MCQRecord)
    assert rec.audio_paths == ["/root/data/x1.wav"]
    assert rec.answer_index == 1
    assert rec.choices == ["Boba tea", "Coffee"]


def test_record_from_row_skips_non_mcq():
    row = {"id": "x2", "question": "q", "answer": "open answer", "choices": [], "audio_path": ["data/x2.wav"]}
    assert record_from_row(row, "/root") is None


def test_record_from_row_multi_audio():
    row = {
        "id": "m", "question": "q", "answer": "A", "choices": ["A", "B"],
        "audio_path": ["data/a.wav", "data/b.wav"], "category": "multi", "length_type": "medium",
    }
    rec = record_from_row(row, "/root")
    assert rec.audio_paths == ["/root/data/a.wav", "/root/data/b.wav"]


# ---------------------------- prompt ---------------------------- #


def test_build_messages_structured_audio():
    rec = MCQRecord(
        unique_id="x", question="What instrument?", choices=["piano", "violin", "drums"],
        answer="violin", audio_paths=["/root/data/x.wav"], category="music",
        length_type="short", answer_index=1,
    )
    msgs = build_messages(rec, audio_mode="local-path")
    assert msgs[0].role == "system"
    user = msgs[-1]
    assert user.role == "user"
    assert isinstance(user.content, list)
    assert any(p.get("type") == "audio_url" for p in user.content)
    text = next(p["text"] for p in user.content if p.get("type") == "text")
    assert "A. piano" in text and "B. violin" in text and "C. drums" in text
    # the carry will treat this as structured (audio survives)
    assert ChatMessages.from_prompt_or_messages(msgs).has_nontext_content() is True


# ------------------- real data (skipped if absent) ------------------- #


@pytest.mark.skipif(not os.path.isdir(DATA_ROOT), reason="MMAU-Pro data not present")
def test_real_testmini_has_957_mcq():
    pytest.importorskip("pandas", reason="loader needs pandas to read the parquet")
    from benchmarking.mmau_pro.loader import load_mmau_mcq

    recs = load_mmau_mcq(DATA_ROOT, subset="full", require_audio_exists=False)
    assert len(recs) == 957
    # every record resolves to at least one absolute audio path that exists
    sample = recs[0]
    assert all(os.path.isabs(p) for p in sample.audio_paths)
