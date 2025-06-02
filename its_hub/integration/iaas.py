# Inference-as-a-Service (IaaS) integration

import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import time
import uuid
import uvicorn
import click

from ..lms import OpenAICompatibleLanguageModel, StepGeneration
from ..algorithms import BestOfN, ParticleFiltering

app = FastAPI()

# global objects
LM_DICT = {}
SCALING_ALG = None

class ConfigRequest(BaseModel):
    endpoint: str
    api_key: str
    model: str
    alg: str
    step_token: Optional[str] = None
    stop_token: Optional[str] = None
    rm_name: str
    rm_device: str
    rm_agg_method: Optional[str] = None

@app.post("/configure")
async def config_service(request: ConfigRequest):
    from its_hub.integration.reward_hub import LocalVllmProcessRewardModel, AggregationMethod

    global LM_DICT, SCALING_ALG

    print(f"{request=}")
    
    # configure language model
    LM = OpenAICompatibleLanguageModel(
        endpoint=request.endpoint,
        api_key=request.api_key,
        model_name=request.model,
        # is_async=True,
    )
    LM_DICT[request.model] = LM
    
    # configure scaling algorithm
    if request.alg == "particle-filtering":
        sg = StepGeneration(
            request.step_token, 
            50, 
            request.stop_token, 
            temperature=0.001, 
            include_stop_str_in_output=True,
            temperature_switch=(0.8, "<boi>", "<eoi>"), # switch to 0.8 when outputing the thinking tokens
        )
        prm = LocalVllmProcessRewardModel(
            model_name=request.rm_name, 
            device=request.rm_device, 
            aggregation_method=AggregationMethod(request.rm_agg_method)
        )
        SCALING_ALG = ParticleFiltering(sg, prm)
    if request.alg == "best-of-n":
        prm = LocalVllmProcessRewardModel(
            model_name=request.rm_name, 
            device=request.rm_device, 
            aggregation_method=AggregationMethod("model")
        )
        orm = prm
        SCALING_ALG = BestOfN(orm)
    
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
    temperature: Optional[float] = None
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
    try:
        lm = LM_DICT[request.model]
    except KeyError:
        raise HTTPException(
            status_code=404, 
            detail=f"Model {request.model} not found. Please configure the model first or use an avaiable from {LM_DICT.keys()}."
        )
    lm.temperature = request.temperature
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
@click.option('--dev', is_flag=True, help='run in development mode')
def main(host: str, port: int, dev: bool):
    print(f"starting IaaS API server on {host}:{port}")
    if dev:
        uvicorn.run("its_hub.integration.iaas:app", host=host, port=port, reload=True)
    else:
        uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
