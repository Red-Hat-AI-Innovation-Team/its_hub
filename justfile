[private]
@default:
    just --list

serve-llama:
    #!/usr/bin/env bash
    CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Llama-3.2-1B-Instruct \
        --dtype float16 \
        --port 8100 \
        --hf-overrides '{"max_position_embeddings": 16384}' \
        --max-model-len 12288

serve-qwen:
    #!/usr/bin/env bash
    CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-Math-1.5B-Instruct \
        --dtype float16 \
        --port 8100 \
        --hf-overrides '{"max_position_embeddings": 16384}' \
        --max-model-len 12288

serve-phi:
    #!/usr/bin/env bash
    CUDA_VISIBLE_DEVICES=0 vllm serve microsoft/Phi-4-mini-instruct \
        --dtype float16 \
        --port 8100 \
        --hf-overrides '{"max_position_embeddings": 16384}' \
        --max-model-len 12288

exp-llama budgets="32":
    #!/usr/bin/env bash
    python scripts/benchmark.py \
        --benchmark aime-2024 \
        --model_name meta-llama/Llama-3.2-1B-Instruct \
        --endpoint http://localhost:8100/v1 \
        --rm_name Qwen/Qwen2.5-Math-PRM-7B \
        --rm_device cuda:1 \
        --rm_agg_method model \
        --alg particle-filtering \
        --budgets {{budgets}} \
        --output_dir results \
        --shuffle_seed 42 \
        --does_eval \
        --subset :30 \
        --temperature 0.8 \
        --max_tokens 2048 \
        --eval_expected_pass_at_one \
        --max_concurrency {{budgets}} \
        --is_async

exp-qwen budgets="32":
    #!/usr/bin/env bash
    python scripts/benchmark.py \
        --benchmark aime-2024 \
        --model_name Qwen/Qwen2.5-Math-1.5B-Instruct \
        --endpoint http://localhost:8100/v1 \
        --rm_name Qwen/Qwen2.5-Math-PRM-7B \
        --rm_device cuda:1 \
        --rm_agg_method model \
        --alg particle-filtering \
        --budgets {{budgets}} \
        --output_dir results \
        --shuffle_seed 42 \
        --does_eval \
        --subset :30 \
        --temperature 0.8 \
        --max_tokens 2048 \
        --eval_expected_pass_at_one \
        --max_concurrency {{budgets}} \
        --is_async

exp-phi budgets="32":
    #!/usr/bin/env bash
    python scripts/benchmark.py \
        --benchmark aime-2024 \
        --model_name microsoft/Phi-4-mini-instruct \
        --endpoint http://localhost:8100/v1 \
        --rm_name Qwen/Qwen2.5-Math-PRM-7B \
        --rm_device cuda:1 \
        --rm_agg_method model \
        --alg particle-filtering \
        --budgets {{budgets}} \
        --output_dir results \
        --shuffle_seed 42 \
        --does_eval \
        --subset :30 \
        --temperature 0.8 \
        --max_tokens 2048 \
        --eval_expected_pass_at_one \
        --max_concurrency {{budgets}} \
        --is_async

iaas-dev:
    #!/usr/bin/env bash
    CUDA_VISIBLE_DEVICES=1 its-iaas \
        --host 0.0.0.0 \
        --port 8108 \
        --dev

iaas-dev-configure-pf:
    #!/usr/bin/env bash
    curl -X POST http://localhost:8108/configure \
        -H "Content-Type: application/json" \
        -d '{"endpoint": "http://localhost:8100/v1", "api_key": "NO_API_KEY", "model": "microsoft/Phi-4-mini-instruct", "alg": "particle-filtering", "step_token": "\n", "stop_token": "<|end|>", "rm_name": "Qwen/Qwen2.5-Math-PRM-7B", "rm_device": "cuda:0", "rm_agg_method": "model"}'

iaas-dev-configure-bon:
    #!/usr/bin/env bash
    curl -X POST http://localhost:8108/configure \
        -H "Content-Type: application/json" \
        -d '{"endpoint": "http://localhost:8100/v1", "api_key": "NO_API_KEY", "model": "microsoft/Phi-4-mini-instruct", "alg": "best-of-n", "rm_name": "Qwen/Qwen2.5-Math-PRM-7B", "rm_device": "cuda:0", "rm_agg_method": "model"}'

iaas-dev-chat prompt="How are you?":
    #!/usr/bin/env bash
    prompt="{{prompt}}"
    echo "raw"
    curl -X POST http://localhost:8100/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"microsoft/Phi-4-mini-instruct\", \"messages\": [{\"role\": \"user\", \"content\": \"$prompt\"}]}"
    echo "\n"
    echo "its"
    curl -X POST http://localhost:8108/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"microsoft/Phi-4-mini-instruct\", \"messages\": [{\"role\": \"user\", \"content\": \"$prompt\"}], \"budget\": 8}"
