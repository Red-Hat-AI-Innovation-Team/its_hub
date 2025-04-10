from .base import AbstractOutcomeRewardModel, AbstractProcessRewardModel


# TODO implement local VLLM-based outcome reward model
class LocalVLLMOutcomeRewardModel(AbstractOutcomeRewardModel):
    pass

# TODO(GX) implement remote VLLM-based outcome reward model
class RemoteVLLMOutcomeRewardModel(AbstractOutcomeRewardModel):
    pass

# TODO implement transformers-based outcome reward model
class TransformersOutcomeRewardModel(AbstractOutcomeRewardModel):
    pass

# TODO implement local VLLM-based process reward model
class LocalVLLMProcessRewardModel(AbstractProcessRewardModel):
    pass

# TODO(GX) implement remote VLLM-based process reward model
class RemoteVLLMProcessRewardModel(AbstractProcessRewardModel):
    pass

# TODO implement transformers-based process reward model
class TransformersProcessRewardModel(AbstractProcessRewardModel):
    pass

class EnsembleOutcomeRewardModel(AbstractOutcomeRewardModel):
    pass

class EnsembleProcessRewardModel(AbstractProcessRewardModel):
    pass

