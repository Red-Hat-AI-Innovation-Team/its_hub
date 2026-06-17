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


def _is_option_token(match: re.Match, text: str) -> bool:
    """Reject 'A'/'I' captures that are English words rather than option letters.

    A letter other than A/I is always plausible. For A/I, require that the token
    is NOT followed by a prose word (e.g. "I believe", "A piano") — i.e. the next
    non-space character must be punctuation or end-of-text/line.
    """
    letter = match.group(1).upper()
    if letter not in ("A", "I"):
        return True
    rest = text[match.end() :]
    return re.match(r"\s*([).,:;!?\]]|$|\n)", rest) is not None


def extract_letter(text: str, num_choices: int) -> int | None:
    """Parse the chosen option letter from model output → choice index, or None."""
    if not text:
        return None
    valid = LETTERS[:num_choices]
    # 0) explicit \boxed{X} — an unambiguous final-answer marker (take the last).
    #    No _is_option_token guard: the braces already disambiguate (and "}" is not
    #    in that helper's trailing-punctuation set, so it would wrongly reject A/I).
    for m in reversed(
        list(re.finditer(r"\\boxed\s*\{\s*\(?([A-K])", text, re.IGNORECASE))
    ):
        c = m.group(1).upper()
        if c in valid:
            return valid.index(c)
    # 1) explicit "Answer: X" (take the last such marker)
    for m in reversed(
        list(re.finditer(r"answer\s*[:\-]?\s*\(?([A-K])\b", text, re.IGNORECASE))
    ):
        c = m.group(1).upper()
        if c in valid and _is_option_token(m, text):
            return valid.index(c)
    # 2) otherwise the last standalone UPPERCASE valid letter token
    #    (uppercase only, so natural-language "a"/"i" aren't mistaken for options;
    #    A/I additionally require option-like trailing context, see _is_option_token)
    for m in reversed(list(re.finditer(r"\b([A-K])\b", text))):
        c = m.group(1)
        if c in valid and _is_option_token(m, text):
            return valid.index(c)
    return None


def predicted_index(text: str, choices: list[str], text_threshold: float = 0.6) -> int | None:
    """The model's chosen choice index: by letter, else by choice-text match."""
    idx = extract_letter(text, len(choices))
    if idx is not None:
        return idx
    nt = normalize(text)
    # prefer the LONGEST choice text contained in the output, so an option that
    # is a substring of another ("dog" vs "dog barking") can't shadow it
    norm_choices = [(i, normalize(c)) for i, c in enumerate(choices)]
    contained = [(i, nc) for i, nc in norm_choices if nc and nc in nt]
    if contained:
        return max(contained, key=lambda pair: len(pair[1]))[0]
    best, best_r = None, 0.0
    for i, nc in norm_choices:
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
