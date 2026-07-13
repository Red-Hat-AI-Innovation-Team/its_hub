"""Unit tests for StepGeneration's stop_regex and stop_on_repeat trajectory-stop rules."""

from its_hub.core.lms.step_generation import (
    StepGeneration,
    _is_repeat_step,
    _normalize_for_repeat,
)

LETTER_ANSWER = r"Answer:\s*(\\boxed\{)?\(?[A-K]\b"


class TestStopRegex:
    def _sg(self):
        return StepGeneration(step_token="\n\n", stop_token="Answer:", max_steps=6,
                              stop_regex=LETTER_ANSWER)

    def test_letter_final_answer_stops(self):
        sg = self._sg()
        assert sg._step_is_final("Answer: D", [])
        assert sg._step_is_final("Answer: D. Decrease", [])
        assert sg._step_is_final("Final Answer: \\boxed{B}", [])
        assert sg._step_is_final("Answer: (C)", [])

    def test_prose_sub_answer_does_not_stop(self):
        sg = self._sg()
        # These are P5-style sub-answers that the plain 'Answer:' substring
        # rule wrongly treats as terminal.
        assert not sg._step_is_final(
            "Sub-question 1: How many sounds?\nAnswer: There are two foley sounds.", [])
        assert not sg._step_is_final("Answer: Yes", [])
        assert not sg._step_is_final("Answer: No, it is static noise.", [])

    def test_regex_overrides_substring_rule(self):
        # same text, substring rule stops, regex rule does not
        plain = StepGeneration(step_token="\n\n", stop_token="Answer:", max_steps=6)
        assert plain._step_is_final("Answer: There are two.", [])
        assert not self._sg()._step_is_final("Answer: There are two.", [])


class TestRepeatGuard:
    def test_digit_incremented_repeat_detected(self):
        prev = ["Sub-question 8: How many sound effects are there in the audio?\n"
                "Answer: There are two sound effects in the audio."]
        nxt = ("Sub-question 12: How many sound effects are there in the audio?\n"
               "Answer: There are two sound effects in the audio.")
        assert _is_repeat_step(nxt, prev)

    def test_repeat_against_any_earlier_step(self):
        prev = ["Sub-question 1: What kind of sound?\nAnswer: It is static noise.",
                "Sub-question 2: Is it constant?\nAnswer: Yes, it is constant noise here."]
        nxt = "Sub-question 5: What kind of sound?\nAnswer: It is static noise."
        assert _is_repeat_step(nxt, prev)

    def test_distinct_step_not_flagged(self):
        prev = ["Sub-question 1: What is the tonality of the verse section of the song?"]
        nxt = "Sub-question 2: What is the tonality of the chorus section of the song?"
        assert not _is_repeat_step(nxt, prev)

    def test_short_steps_ignored(self):
        assert not _is_repeat_step("Answer: D", ["Answer: D"])
        assert not _is_repeat_step("", [""])

    def test_wired_into_step_is_final(self):
        sg = StepGeneration(step_token="\n\n", stop_token="Answer:", max_steps=6,
                            stop_regex=LETTER_ANSWER, stop_on_repeat=True)
        prev = ["Sub-question 3: How many sound design elements are there in the audio?\n"
                "Answer: There are two sound design elements."]
        loop = ("Sub-question 7: How many sound design elements are there in the audio?\n"
                "Answer: There are two sound design elements.")
        assert sg._step_is_final(loop, prev)          # repeat -> stop
        assert not sg._step_is_final(loop, [])        # first occurrence -> continue

    def test_normalization(self):
        assert _normalize_for_repeat("Sub-question 12:  FOO bar") == \
               _normalize_for_repeat("sub-question 3: foo   BAR")
