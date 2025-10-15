from reward_hub.vllm.reward import VllmProcessRewardModel

model = VllmProcessRewardModel(model_name="Qwen/Qwen2.5-Math-PRM-7B", device="cpu")

messages = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing well, thank you!"},
]

result = model.score(messages=messages, aggregation_method="model", return_full_prm_result=False)
print(result)