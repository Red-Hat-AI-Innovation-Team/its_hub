from typing import List, Union

import numpy as np

from reward_hub.base import AggregationMethod
from reward_hub import AutoRM
from reward_hub.drsow import DrSowConfig

from ..base import AbstractOutcomeRewardModel, AbstractProcessRewardModel

def sigmoid(x):
    # use numpy's clip to avoid overflow in the exponential
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88.0, 88.0)))

class LocalVllmProcessRewardModel(AbstractProcessRewardModel):
    def __init__(self, model_name: str, device: str, aggregation_method: AggregationMethod):
        from reward_hub.vllm.reward import VllmProcessRewardModel
        if "drsow" in model_name:
            strong_model_name, weak_model_name = model_name.split("+")[1:]

            drsow_config = DrSowConfig(
                strong_model_name=strong_model_name,
                strong_port=8001,
                weak_model_name=weak_model_name,
                weak_port=8002, 
            )

            self.model = AutoRM.load("drsow", load_method="openai", drsow_config=drsow_config)
            self.score_scale = 1.0 / 0.03
            self.func = sigmoid
        else:
            self.model = VllmProcessRewardModel(
                model_name=model_name, device=device
            )
            self.score_scale = 1.0
            self.func = lambda x: x
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
        res = [self.func(r) * self.score_scale for r in res]
        if is_single_prompt:
            return res[0]
        else:
            return res