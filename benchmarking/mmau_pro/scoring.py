"""Answer extraction + MCQ scoring for MMAU-Pro (pure functions, offline-testable).

MMAU-Pro `answer` is the choice TEXT (not a letter), and `choices` is a list of up
to ~11 options. We present lettered choices (A, B, ...), parse the model's chosen
letter, and compare the selected choice index to the precomputed answer index.
A normalized/fuzzy text fallback handles outputs that give the answer text instead
of a clean letter, and the ~23/957 cases where `answer` is not verbatim in `choices`.
"""

import re
from difflib import SequenceMatcher

LETTERS = "ABCDEFGHIJK"  # supports up to 11 choices (max seen in MMAU-Pro test-mini)


def normalize(s: str | None) -> str:
    """Lowercase, drop punctuation and articles, collapse whitespace."""
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def match_answer_index(answer: str, choices: list[str], threshold: float = 0.85) -> int | None:
    """Map the gold `answer` text to its index in `choices`.

    Exact-normalized match first; then best fuzzy ratio above `threshold`.
    Returns None if nothing matches (caller treats the item as ungradeable).
    """
    if not choices:
        return None
    na = normalize(answer)
    norm_choices = [normalize(c) for c in choices]
    if na in norm_choices:
        return norm_choices.index(na)
    best, best_r = None, 0.0
    for i, nc in enumerate(norm_choices):
        r = SequenceMatcher(None, na, nc).ratio()
        if r > best_r:
            best, best_r = i, r
    return best if best_r >= threshold else None


def extract_letter(text: str, num_choices: int) -> int | None:
    """Parse the chosen option letter from model output → choice index, or None."""
    if not text:
        return None
    valid = LETTERS[:num_choices]
    # 1) explicit "Answer: X" (take the last such marker)
    m = re.findall(r"answer\s*[:\-]?\s*\(?([A-K])\b", text, re.IGNORECASE)
    if m:
        c = m[-1].upper()
        if c in valid:
            return valid.index(c)
    # 2) otherwise the last standalone UPPERCASE valid letter token
    #    (uppercase only, so natural-language "a"/"i" aren't mistaken for options)
    for c in reversed(re.findall(r"\b([A-K])\b", text)):
        if c in valid:
            return valid.index(c)
    return None


def predicted_index(text: str, choices: list[str], text_threshold: float = 0.6) -> int | None:
    """The model's chosen choice index: by letter, else by choice-text match."""
    idx = extract_letter(text, len(choices))
    if idx is not None:
        return idx
    nt = normalize(text)
    best, best_r = None, 0.0
    for i, c in enumerate(choices):
        nc = normalize(c)
        if nc and nc in nt:
            return i
        r = SequenceMatcher(None, nc, nt).ratio()
        if r > best_r:
            best, best_r = i, r
    return best if best_r >= text_threshold else None


def is_correct(text: str, choices: list[str], answer_index: int | None) -> bool | None:
    """True/False if gradeable, None if the item is ungradeable (answer_index is None)."""
    if answer_index is None:
        return None
    pi = predicted_index(text, choices)
    if pi is None:
        return False
    return pi == answer_index
