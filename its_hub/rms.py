import re
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
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

QWEN_PRM_SYSTEM_PROMPT = \
    "Please reason step by step, and put your final answer within \\boxed{}."
QWEN_PRM_REWARD_TOKEN = "<extra_0>"

def make_step_rewards(logits, token_masks):
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels
    
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i] # seq_len, num_labels
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1] # valid_tokens, num_labels
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res

class LocalVLLMProcessRewardModel(AbstractProcessRewardModel):
    def __init__(
        self, 
        model_name: str, 
        system_prompt: str = QWEN_PRM_SYSTEM_PROMPT, 
        reward_token: str = QWEN_PRM_REWARD_TOKEN, 
    ):
        self.model = AutoModel.from_pretrained(
            model_name, 
            # device_map="auto", 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True, 
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.system_prompt = system_prompt
        self.reward_token = reward_token
    
    def score(
        self, 
        questions: list[str], 
        outputs: list[list[str]], 
        outputs_is_single_step: bool = True, 
    ) -> list[list[float]]:
        all_scores = []
        for question, answers in zip(questions, outputs, strict=True):
            all_step_scores = []
            for ans in answers:
                # we assume here that the answers use "\n\n" to separate steps. 
                if outputs_is_single_step:
                    ans = re.sub(r'\n+', '\n', ans)

                steps_list = ans.split("\n\n")
                
                messages = [
                    {"role": "system", 
                     "content": self.system_prompt},
                    {"role": "user", 
                     "content": question},
                    {"role": "assistant", 
                     "content": self.reward_token.join(steps_list) + self.reward_token},
                ]

                # Prepare conversation for scoring
                conversation = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )

                input_ids = self.tokenizer.encode(
                    conversation, 
                    return_tensors="pt", 
                ).to(self.model.device)

                outputs = self.model(input_ids=input_ids)

                # get the step scores
                step_sep_id = self.tokenizer.encode("<extra_0>")[0]
                token_masks = (input_ids == step_sep_id)
                step_scores = make_step_rewards(outputs[0], token_masks)

                # make the scores cumulative through multiplication
                # step_scores = [math.prod(step_scores[:i+1]) for i in range(len(step_scores))]

                all_step_scores.extend(step_scores)

            all_scores.append(all_step_scores)

        return all_scores


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

