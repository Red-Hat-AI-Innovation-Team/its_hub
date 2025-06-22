from reward_hub.base import AggregationMethod

from its_hub.base import AbstractProcessRewardModel


class LocalVllmProcessRewardModel(AbstractProcessRewardModel):
    def __init__(
        self, model_name: str, device: str, aggregation_method: AggregationMethod
    ):
        from reward_hub.vllm.reward import VllmProcessRewardModel

        self.model = VllmProcessRewardModel(model_name=model_name, device=device)
        self.aggregation_method = aggregation_method

    def score(self, prompt: str, response_or_responses: str | list[str]) -> float:
        is_single_response = isinstance(response_or_responses, str)
        messages = [
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": r}]
            for r in (
                [response_or_responses] if is_single_response else response_or_responses
            )
        ]
        res = self.model.score(
            messages=messages,
            aggregation_method=self.aggregation_method,
            return_full_prm_result=False,
        )
        if is_single_response:
            return res[0]
        else:
            return res
