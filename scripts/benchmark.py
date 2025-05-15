from typing import Union, List
from enum import Enum
import os
import re
import click
import datasets
import numpy as np
import pandas as pd
from tqdm import tqdm
import math_verify

from its_hub.lms import OpenAICompatibleLanguageModel
from its_hub.algorithms import SelfConsistency, BeamSearch, ParticleFiltering, StepGeneration
from its_hub.algorithms.particle_gibbs import _softmax
from its_hub.utils import SAL_STEP_BY_STEP_SYSTEM_PROMPT, QWEN_SYSTEM_PROMPT
from its_hub.integration.reward_hub import AggregationMethod, LocalVllmProcessRewardModel

class BenchmarkDataset(Enum):
    MATH500 = "math500"
    AIME_2024 = "aime-2024"

def load_benchmark_dataset(dataset: BenchmarkDataset):
    if dataset == BenchmarkDataset.MATH500:
        ds = datasets.load_dataset("HuggingFaceH4/MATH-500")["test"]
    elif dataset == BenchmarkDataset.AIME_2024:
        ds = datasets.load_dataset("Maxwell-Jia/AIME_2024")["train"]
        old_column_names = ds.column_names
        ds = ds.map(lambda x: {k.lower(): v for k, v in x.items()})
        # use existing id as unique_id
        ds = ds.rename_column('id', 'unique_id')
        # convert answer to string type
        ds = ds.cast_column('answer', datasets.Value('string'))
        # remove old columns
        ds = ds.remove_columns(old_column_names)
    # add unique_id if it doesn't exist
    if 'unique_id' not in ds.column_names:
        ds = ds.map(lambda _, idx: {'unique_id': idx}, with_indices=True)
    return ds
    
class ScalingAlgorithm(Enum):
    SELF_CONSISTENCY = "self-consistency"
    BEAM_SEARCH = "beam-search"
    PARTICLE_FILTERING = "particle-filtering"

def _extract_boxed(s: str) -> str:
    # find all occurrences of \boxed{...}
    boxed_matches = re.findall(r'\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}', s)
    # return the last match if any were found
    return boxed_matches[-1] if boxed_matches else ""

def init_algorithm(alg: ScalingAlgorithm, model_name: str, rm_name: str, rm_device: str, rm_agg_method: AggregationMethod):
    step_token = "\n\n##" if "llama" in model_name.lower() else "\n\n"
    if alg == ScalingAlgorithm.SELF_CONSISTENCY:
        return SelfConsistency(_extract_boxed)
    elif alg == ScalingAlgorithm.BEAM_SEARCH:
        sg = StepGeneration(step_token, 50, "\\boxed")
        prm = LocalVllmProcessRewardModel(
            model_name=rm_name, device=rm_device, aggregation_method=rm_agg_method
        )
        return BeamSearch(sg, prm, beam_width=4)
    elif alg == ScalingAlgorithm.PARTICLE_FILTERING:
        sg = StepGeneration(step_token, 50, "\\boxed")
        prm = LocalVllmProcessRewardModel(
            model_name=rm_name, device=rm_device, aggregation_method=rm_agg_method
        )
        return ParticleFiltering(sg, prm)

def display_results(df: pd.DataFrame):
    if len(df) == 0:
        print("no results to display")
        return
    # print accuracy per budget using groupby and mean
    accuracy_by_budget = df.groupby("budget")["correct"].agg(["mean", "count"])
    for n, (accuracy, count) in accuracy_by_budget.iterrows():
        print(f"budget={n:3d}: accuracy={accuracy:.4f} ({int(accuracy * count):2d}/{int(count):2d})")

@click.command()
@click.option("--benchmark", type=click.Choice([e.value for e in BenchmarkDataset]), required=True, 
              callback=lambda ctx, param, value: BenchmarkDataset(value),
              help="dataset to use for benchmarking")
@click.option("--model_name", type=str, required=True, help="model to inference-time scale")
@click.option("--is_async", is_flag=True, default=False, help="whether to use async mode")
@click.option("--max_tokens", type=int, default=None, help="max tokens to use for inference-time scaling")
@click.option("--temperature", type=float, default=None, help="temperature to use for inference-time scaling")
@click.option("--max_concurrency", type=int, default=8, help="max concurrency to use for inference-time scaling")
@click.option("--endpoint", type=str, help="endpoint to use for inference-time scaling")
@click.option("--api_key", type=str, default="NO_API_KEY", help="api key to use for inference-time scaling")
@click.option("--rm_name", type=str, default="Qwen/Qwen2.5-Math-PRM-7B", help="name of reward model to use")
@click.option("--rm_device", type=str, default="cpu", help="device to use for reward model")
@click.option("--rm_agg_method", type=click.Choice([e.value for e in AggregationMethod]), default="model", 
              callback=lambda ctx, param, value: AggregationMethod(value),
              help="aggregation method to use for reward model")
@click.option("--alg", type=click.Choice([e.value for e in ScalingAlgorithm]), required=True, 
              callback=lambda ctx, param, value: ScalingAlgorithm(value),
              help="algorithm to use for inference-time scaling")
@click.option("--subset", type=str, default=None, help="subset of dataset to use, in python slice syntax (e.g. ':10', '5:', '5:10')")
@click.option("--budgets", type=str, default="1,2,4,8", 
              callback=lambda ctx, param, value: [int(b) for b in value.split(",")],
              help="comma-separated list of budgets to use for inference-time scaling")
@click.option("--output_dir", type=str, default="results", help="directory to save results to")
@click.option("--shuffle_seed", type=int, default=None, help="random seed to use for shuffling")
@click.option("--force_run", is_flag=True, default=False, help="whether to force re-running")
@click.option("--does_eval", is_flag=True, default=False, help="whether to evaluate the results")
@click.option("--eval_expected_pass_at_one", is_flag=True, default=False, help="whether to evaluate expected pass at one")
@click.option("--display_only", is_flag=True, default=False, help="whether to show only the results")
def main(
    benchmark: BenchmarkDataset, 
    model_name: str, 
    is_async: bool,
    max_tokens: int,
    temperature: float,
    max_concurrency: int,
    endpoint: str, 
    api_key: str, 
    rm_name: str,
    rm_device: str,
    rm_agg_method: AggregationMethod,
    alg: ScalingAlgorithm, 
    subset: str, 
    budgets: list, 
    output_dir: str,
    shuffle_seed: int,
    force_run: bool,
    does_eval: bool,
    eval_expected_pass_at_one: bool,
    display_only: bool,
):
    # print all arguments using click context
    ctx = click.get_current_context()
    print("running with arguments:")
    for param_name, param_value in ctx.params.items():
        print(f"  {param_name}: {param_value}")

    if eval_expected_pass_at_one:
        assert alg == ScalingAlgorithm.PARTICLE_FILTERING, "expected pass at one is only supported for particle filtering"

    print("loading existing results...")
    model_name_dashed = model_name.replace("/", "-")
    if alg == ScalingAlgorithm.BEAM_SEARCH or alg == ScalingAlgorithm.PARTICLE_FILTERING:
        rm_name_dashed = rm_name.replace("/", "-")
        alg_str = f"{alg.value}-{rm_name_dashed}-{rm_agg_method.value}"
    else:
        alg_str = alg.value
    output_file = os.path.join(output_dir, f"{model_name_dashed}-{alg_str}-{benchmark.value}.jsonl")
    if os.path.exists(output_file):
        df_existing = pd.read_json(output_file, orient='records', lines=True)
        print(f"loaded {len(df_existing)} existing results from {output_file}")
    else:
        df_existing = pd.DataFrame()
    
    if display_only:
        display_results(df_existing)
        return

    print("loading benchmark dataset...")
    dataset = load_benchmark_dataset(benchmark)

    if shuffle_seed is not None:
        dataset = dataset.shuffle(seed=shuffle_seed)
    
    # apply subset if specified
    if subset is not None:
        try:
            # parse the slice syntax
            if ':' in subset:
                parts = subset.split(':')
                if len(parts) == 2:
                    start = int(parts[0]) if parts[0] else None
                    end = int(parts[1]) if parts[1] else None
                    dataset = dataset.select(range(start if start is not None else 0, 
                                                   end if end is not None else len(dataset)))
            else:
                # single index
                dataset = dataset.select([int(subset)])
            print(f"using subset of dataset: {len(dataset)} examples")
        except ValueError:
            print(f"invalid subset format: {subset}, using full dataset")

    print("creating language model...")
    if endpoint is not None:
        lm = OpenAICompatibleLanguageModel(
            endpoint=endpoint, 
            api_key=api_key, 
            model_name=model_name, 
            system_prompt=QWEN_SYSTEM_PROMPT if "qwen" in model_name.lower() else SAL_STEP_BY_STEP_SYSTEM_PROMPT, 
            is_async=is_async,
            temperature=temperature,
            max_tokens=max_tokens,
            max_concurrency=max_concurrency,
        )

    print("initializing algorithm...")
    scaling_alg = init_algorithm(alg, model_name, rm_name, rm_device, rm_agg_method)

    # ensure output directory exists
    if not os.path.exists(output_dir):
        print(f"creating output directory: {output_dir}")
        os.makedirs(output_dir)
    
    print(f"running inference-time scaling for {budgets=}...")
    rows = []
    try:
        for n in tqdm(budgets):
            for x in dataset:
                y_full = None
                y = None
                if not force_run and len(df_existing) > 0:
                    # only skip if both the unique_id and budget matches
                    match = (df_existing["unique_id"] == x["unique_id"]) & \
                            (df_existing["budget"] == n)
                    if match.any():
                        assert match.sum() == 1, f"expected exactly one match, got {match.sum()}"
                        if eval_expected_pass_at_one:
                            y_full = {
                                "responses": df_existing.loc[match, "responses"].values[0],
                                "log_probs": df_existing.loc[match, "log_probs"].values[0],
                            }
                        else:
                            y = df_existing.loc[match, "response"].values[0]
                if (y_full is None if eval_expected_pass_at_one else y is None):
                    try:
                        if eval_expected_pass_at_one:
                            y_full = scaling_alg.infer(lm, x["problem"], n, return_response_only=False)
                            y_full = {
                                "responses": y_full.responses_lst[-1],
                                "log_probs": y_full.log_weights_lst[-1],
                            }
                        else:
                            y = scaling_alg.infer(lm, x["problem"], n)
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        print(f"error scaling example {x['unique_id']}: {e}")
                        continue
                if eval_expected_pass_at_one:
                    row = {
                        "unique_id": x["unique_id"],
                        "budget": n,
                        "responses": y_full["responses"],
                        "log_probs": y_full["log_probs"],
                        "correct": None,
                    }
                else:
                    row = {
                        "unique_id": x["unique_id"],
                        "budget": n,
                        "response": y,
                        "correct": None,
                    }
                if does_eval:
                    if eval_expected_pass_at_one:
                        c = [math_verify.verify(math_verify.parse(x["answer"]), math_verify.parse(y)) 
                             for y in row["responses"]]
                        p = _softmax(row["log_probs"])
                        row["correct"] = np.dot(p, c)
                    else:
                        row["correct"] = math_verify.verify(
                            math_verify.parse(x["answer"]), math_verify.parse(row["response"])
                        )
                rows.append(row)
    except KeyboardInterrupt:
        print("\nkeyboard interrupt detected, saving partial results")
    
    # save results to jsonl file using pandas
    print(f"saving results to {output_file}...")
    df = pd.concat([df_existing, pd.DataFrame(rows)])
    # deduplicate rows with the same unique_id and budget, keeping the updated correctness
    df = df.drop_duplicates(subset=['unique_id', 'budget'], keep='last')
    
    display_results(df)

    df.to_json(output_file, orient='records', lines=True)

if __name__ == "__main__":
    main()
