"""
self_consistency_demo.py — watch Self-Consistency vote on extracted answers.

Runnable companion to documentation/05-self-consistency-and-best-of-n.md. No GPU / server / key.

    /home/exx/miniconda3/envs/epf/bin/python documentation/snippets/self_consistency_demo.py
    # or:  conda run -n epf python documentation/snippets/self_consistency_demo.py

Uses the REAL SelfConsistency algorithm with a regex projection that extracts the \\boxed{...}
answer, driven by a mock LM that returns a fixed set of solutions. The crowd should agree on 42.
"""

import random

from its_hub.core.algorithms.self_consistency import (
    SelfConsistency,
    create_regex_projection_function,
)
from its_hub.api import AbstractLanguageModel


class MockLM(AbstractLanguageModel):
    """Returns preset answers in order; agenerate_single is what the orchestrator calls."""

    def __init__(self, answers: list[str]) -> None:
        self.answers = answers
        self.i = 0

    async def agenerate_single(self, messages, **kwargs) -> dict:
        ans = self.answers[self.i % len(self.answers)]
        self.i += 1
        return {"role": "assistant", "content": ans}

    async def agenerate(self, messages, **kwargs):
        if isinstance(messages, list) and messages and isinstance(messages[0], list):
            return [await self.agenerate_single(m) for m in messages]
        return await self.agenerate_single(messages)


def main() -> None:
    random.seed(0)  # deterministic tie-breaking

    # Four sampled solutions; three reach 42, one reaches 7.
    answers = [
        "First add the parts ... so the answer is \\boxed{42}.",
        "By symmetry the total is \\boxed{42}.",
        "A different route gives \\boxed{7}.",
        "Cross-checking, again \\boxed{42}.",
    ]

    # Project each response down to its boxed answer for voting.
    boxed = create_regex_projection_function(r"\\boxed\{([^}]+)\}")
    sc = SelfConsistency(boxed)

    result = sc.infer(MockLM(answers), "What is the answer?", budget=4, return_response_only=False)

    print("=" * 60)
    print("Self-Consistency vote (projection = boxed answer)")
    print("=" * 60)
    print(f"  votes (response_counts): {dict(result.response_counts)}")
    print(f"  selected_index         : {result.selected_index}")
    print(f"  the_one.content        : {result.the_one['content']!r}")
    print("\n  The most common projected answer ('42') wins; the winning ORIGINAL")
    print("  response (not the projection) is returned as the_one.")


if __name__ == "__main__":
    main()
