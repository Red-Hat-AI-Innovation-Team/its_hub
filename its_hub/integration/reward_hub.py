from typing import List, Union

from reward_hub.base import AggregationMethod

from ..base import AbstractOutcomeRewardModel, AbstractProcessRewardModel

class LocalVllmProcessRewardModel(AbstractProcessRewardModel):
    def __init__(self, model_name: str, device: str, aggregation_method: AggregationMethod):
        from reward_hub.vllm.reward import VllmProcessRewardModel

        self.model = VllmProcessRewardModel(
            model_name=model_name, device=device
        )
        self.aggregation_method = aggregation_method

    def score(self, prompt: str, steps: Union[List[str], List[List[str]]]) -> float:
        is_single_prompt = isinstance(steps[0], str)
        messages = [
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": "\n\n".join(s)}]
            for s in ([steps] if is_single_prompt else steps)
        ]
        res = self.model.score(
            messages=messages,
            aggregation_method=self.aggregation_method,
            return_full_prm_result=False,
        )
        if is_single_prompt:
            return res[0]
        else:
            return res