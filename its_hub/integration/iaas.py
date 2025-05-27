# Inference-as-a-Service (IaaS) integration

import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import time
import uuid
import uvicorn
import click

from ..lms import OpenAICompatibleLanguageModel
from ..algorithms import ParticleFiltering, StepGeneration

app = FastAPI()

# global objects
LM_DICT = {}
SCALING_ALG = None

class ConfigRequest(BaseModel):
    endpoint: str
    api_key: str
    model: str
    alg: str
    rm_name: str
    rm_device: str
    rm_agg_method: str

@app.post("/configure")
async def config_service(request: ConfigRequest):
    from its_hub.integration.reward_hub import LocalVllmProcessRewardModel, AggregationMethod

    global LM_DICT, SCALING_ALG
    
    # configure language model
    LM = OpenAICompatibleLanguageModel(
        endpoint=request.endpoint,
        api_key=request.api_key,
        model_name=request.model,
    )
    LM_DICT[request.model] = LM
    
    # configure scaling algorithm
    sg = StepGeneration("\n", 50, "<|end|>", temperature=0.2, include_stop_str_in_output=True)
    prm = LocalVllmProcessRewardModel(
        model_name=request.rm_name, 
        device=request.rm_device, 
        aggregation_method=AggregationMethod(request.rm_agg_method)
    )
    SCALING_ALG = ParticleFiltering(sg, prm)
    
    return {"status": "success", 
            "message": "initialized language model and inference-time scaling algorithm"}

@app.get("/v1/models")
async def models():
    return {"data": [{"id": model, "object": "model", "owned_by": "its" } for model in LM_DICT.keys()]}

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    budget: int = 8
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    assert len(request.messages) in [1, 2] and request.messages[-1].role == "user", \
        "Only one user message with optional system message is supported"
    lm = LM_DICT[request.model]
    if len(request.messages) == 2:
        lm.system_prompt = request.messages[0].content
    else:
        lm.system_prompt = None
    prompt = request.messages[-1].content
    response = SCALING_ALG.infer(lm, prompt, request.budget)
    return ChatCompletionResponse(
        id=("chatcmpl-" + str(uuid.uuid4())),
        created=int(time.time()),
        model=request.model,
        choices=[{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response
            },
            "finish_reason": "stop"
        }],
        usage={
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    )

@click.command()
@click.option('--host', default='127.0.0.1', help='host to bind the server to')
@click.option('--port', default=8000, help='port to bind the server to')
@click.option('--endpoint', required=True, help='endpoint for the language model')
@click.option('--api-key', required=True, help='api key for the language model')
@click.option('--model', required=True, help='model name to use')
@click.option('--alg', default='particle_filtering', help='scaling algorithm to use')
@click.option('--rm-name', required=True, help='reward model name')
@click.option('--rm-device', default='cuda:0', help='device to run reward model on')
@click.option('--rm-agg-method', default='mean', help='reward model aggregation method')
@click.option('--dev', is_flag=True, help='run in development mode')
def main(host: str, port: int, endpoint: str, api_key: str, model: str, 
               alg: str, rm_name: str, rm_device: str, rm_agg_method: str, dev: bool):
    """start the IaaS service with the specified configuration"""
    # configure the service
    config_request = ConfigRequest(
        endpoint=endpoint,
        api_key=api_key,
        model=model,
        alg=alg,
        rm_name=rm_name,
        rm_device=rm_device,
        rm_agg_method=rm_agg_method
    )
    
    # run the initialization
    asyncio.run(config_service(config_request))
    
    # start the server
    print(f"starting IaaS API server on {host}:{port}")
    if dev:
        uvicorn.run("its_hub.integration.iaas:app", host=host, port=port, reload=True)
    else:
        uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
