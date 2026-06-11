"""Build multimodal MCQ prompts (audio + lettered choices) for MMAU-Pro.

Provides the original terse `build_messages` (used by tests / A-B causality) plus a
set of CoT-elicitation prompt builders `build(method, rec, ...)` shared by the
comparison (`cot_compare`) and the runner (`run_mmau`). All builders return a
`list[ChatMessage]` whose user turn carries the audio as structured content, so the
PF/EPF audio carry delivers it to the model at every reasoning step.
"""

from benchmarking.mmau_pro.audio import audio_content_parts
from benchmarking.mmau_pro.scoring import LETTERS
from its_hub.api.types import ChatMessage

MMAU_MCQ_SYSTEM_PROMPT = (
    "You are an expert audio analyst. Listen to the provided audio carefully and "
    "answer the multiple-choice question. Reason briefly step by step, then end your "
    "response with a line of exactly the form 'Answer: <letter>' giving the single "
    "letter of the correct option."
)

_SYS = "You are an expert audio analyst. Listen to the audio carefully."

# CoT-elicitation methods compared in cot_compare; 2/4/7 are the ablation finalists.
METHODS = {
    1: "assistant-prefill CoT",
    2: "zero-shot CoT (user trigger)",
    3: "few-shot CoT",
    4: "plan-and-solve",
    5: "least-to-most",
    6: "describe-then-reason (audio)",
    7: "format-forcing (## Step)",
    8: "anti-shortcut (>=3 steps)",
}


def format_choices(choices: list[str]) -> str:
    return "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))


def build_messages(
    record,
    audio_mode: str = "local-path",
    system_prompt: str | None = MMAU_MCQ_SYSTEM_PROMPT,
) -> list[ChatMessage]:
    """Original terse MCQ prompt: [system?, user(audio... + question + lettered choices)]."""
    parts = audio_content_parts(record.audio_paths, mode=audio_mode)
    text = (
        f"{record.question}\n\nOptions:\n{format_choices(record.choices)}\n\n"
        "Answer with the letter of the correct option."
    )
    parts.append({"type": "text", "text": text})
    messages: list[ChatMessage] = []
    if system_prompt:
        messages.append(ChatMessage(role="system", content=system_prompt))
    messages.append(ChatMessage(role="user", content=parts))
    return messages


def build(method: int, rec, audio_mode: str = "local-path") -> tuple[list[ChatMessage], str]:
    """Return (messages, assistant_seed) for a CoT-elicitation `method` (see METHODS).

    `assistant_seed` is the prefilled assistant content for method 1 (empty otherwise);
    the runner ignores the seed for the ablation methods (2/4/7), which are prompt-only.
    """
    ap = audio_content_parts(rec.audio_paths, mode=audio_mode)
    q, opts = rec.question, format_choices(rec.choices)

    def user(txt):
        return ChatMessage(role="user", content=[*ap, {"type": "text", "text": txt}])

    def sysm(s):
        return ChatMessage(role="system", content=s)

    base = f"Question: {q}\n\nOptions:\n{opts}"
    seed = ""
    if method == 1:  # assistant prefill zero-shot CoT
        msgs = [
            sysm(_SYS + " Reason step by step, then end with a line 'Answer: <letter>'."),
            user(base),
            ChatMessage(role="assistant", content="Let's think step by step."),
        ]
        seed = "Let's think step by step."
    elif method == 2:  # zero-shot CoT, trigger at end of user turn
        msgs = [sysm(_SYS), user(base + "\n\nAnswer: Let's think step by step.")]
    elif method == 3:  # few-shot CoT (one worked text example)
        ex = (
            "Example\nQuestion: Which sound is heard for the shortest duration?\n"
            "Options:\nA. Music\nB. Human voice\nC. Wind\nD. Cat meowing\n"
            "Answer: Let's think step by step. Music and the human voice are sustained; "
            "a cat meow is short; the wind gust is the briefest. So, the answer is C. Wind.\n\n---\n\n"
        )
        msgs = [sysm(_SYS), user(ex + base + "\n\nAnswer: Let's think step by step.")]
    elif method == 4:  # plan-and-solve
        msgs = [sysm(_SYS), user(base + "\n\nFirst devise a plan to answer the question, then carry out the plan step by step, then end with 'Answer: <letter>'.")]
    elif method == 5:  # least-to-most
        msgs = [sysm(_SYS), user(base + "\n\nBreak this into simpler sub-questions, answer each one in order, then end with 'Answer: <letter>'.")]
    elif method == 6:  # describe-then-reason (audio-grounded)
        msgs = [sysm(_SYS), user(base + "\n\nFirst describe what you hear in the audio. Then reason about the question using that description. Then end with 'Answer: <letter>'.")]
    elif method == 7:  # format-forcing numbered steps
        msgs = [
            sysm(_SYS + " Respond ONLY in this format:\n## Step 1: <reasoning>\n\n## Step 2: <reasoning>\n\n## Step 3: <reasoning>\n\nAnswer: <letter>"),
            user(base),
        ]
    elif method == 8:  # anti-shortcut
        msgs = [sysm(_SYS), user(base + "\n\nDo NOT state the answer until you have written at least 3 numbered reasoning steps grounded in the audio. Then end with 'Answer: <letter>'.")]
    else:
        raise ValueError(f"unknown method {method}")
    return msgs, seed
