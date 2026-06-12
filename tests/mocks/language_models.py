"""Mock language models for testing."""

from its_hub import AbstractLanguageModel


class LogprobMockLM(AbstractLanguageModel):
    """Mock LM that emits OpenAI-style `_logprobs` when `logprobs=True`.

    Each generated step gets a (cycled) target mean logprob so different
    particles end up with different self-certainty weights.
    """

    def __init__(self, mean_logprobs=(-0.1, -0.5, -1.0, -0.2), n_tokens=2):
        self.mean_logprobs = list(mean_logprobs)
        self.n_tokens = n_tokens
        self.call_count = 0
        self.saw_logprobs = False
        self.saw_top_logprobs = None
        self.saw_tools = None
        self.saw_tool_choice = None

    def _make_message(self, idx, want_logprobs, want_top):
        base = self.mean_logprobs[idx % len(self.mean_logprobs)]
        msg = {"role": "assistant", "content": f"step{idx}"}
        if want_logprobs:
            toks = []
            for t in range(self.n_tokens):
                entry = {"token": f"tok{t}", "logprob": base}
                if want_top is not None:
                    entry["top_logprobs"] = [
                        {"token": f"tok{t}", "logprob": base},
                        {"token": "other", "logprob": base - 1.0},
                    ]
                toks.append(entry)
            msg["_logprobs"] = {"content": toks}
        return msg

    async def agenerate(
        self,
        messages,
        stop=None,
        max_tokens=None,
        temperature=None,
        include_stop_str_in_output=None,
        tools=None,
        tool_choice=None,
        response_format=None,
        logprobs=False,
        top_logprobs=None,
    ):
        self.saw_logprobs = self.saw_logprobs or bool(logprobs)
        if top_logprobs is not None:
            self.saw_top_logprobs = top_logprobs
        if tools is not None:
            self.saw_tools = tools
        if tool_choice is not None:
            self.saw_tool_choice = tool_choice
        is_batch = (
            isinstance(messages, list)
            and len(messages) > 0
            and isinstance(messages[0], list)
        )
        if is_batch:
            out = []
            for _ in messages:
                out.append(self._make_message(self.call_count, logprobs, top_logprobs))
                self.call_count += 1
            return out
        msg = self._make_message(self.call_count, logprobs, top_logprobs)
        self.call_count += 1
        return msg

    async def agenerate_single(self, messages, **kwargs):
        return await self.agenerate(messages, **kwargs)
