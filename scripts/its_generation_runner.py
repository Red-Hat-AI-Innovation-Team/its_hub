#!/usr/bin/env python3
"""
Generation Script for Inference-Time Scaling
=============================================

A flexible script for running inference-time scaling algorithms on custom datasets.
Captures comprehensive generation results including all responses, scores, and metadata.

Features:
- Support for custom CSV/JSON datasets
- Hierarchical output organization (dataset/model/algorithm/budget)
- Comprehensive result capture with return_response_only=False
- Enhanced progress tracking
- Results added as new columns to original dataset
- Robust resumption logic
"""

from enum import Enum
from pathlib import Path
from typing import Any, Dict

import click
import datasets
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TaskProgressColumn,
    MofNCompleteColumn,
)
from rich import box

from its_hub.algorithms import (
    BeamSearch,
    BestOfN,
    ParticleFiltering,
    SelfConsistency,
    StepGeneration,
)
from its_hub.integration.reward_hub import (
    AggregationMethod,
    LocalVllmProcessRewardModel,
)
from its_hub.lms import OpenAICompatibleLanguageModel
from its_hub.utils import QWEN_SYSTEM_PROMPT, SAL_STEP_BY_STEP_SYSTEM_PROMPT

# Initialize rich console
console = Console()


class ScalingAlgorithm(Enum):
    SELF_CONSISTENCY = "self-consistency"
    BEAM_SEARCH = "beam-search"
    PARTICLE_FILTERING = "particle-filtering"
    BEST_OF_N = "best-of-n"


def load_dataset_from_file(
    dataset_file: str, prompt_column: str, id_column: str = None
) -> pd.DataFrame:
    """Load dataset from local CSV or JSON file."""
    file_path = Path(dataset_file)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    # Load based on file extension
    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(dataset_file)
    elif file_path.suffix.lower() in [".json", ".jsonl"]:
        if file_path.suffix.lower() == ".jsonl":
            df = pd.read_json(dataset_file, lines=True)
        else:
            df = pd.read_json(dataset_file)
    else:
        raise ValueError(
            f"Unsupported file format: {file_path.suffix}. Use .csv, .json, or .jsonl"
        )

    # Validate prompt column exists
    if prompt_column not in df.columns:
        raise ValueError(
            f"Prompt column '{prompt_column}' not found in dataset. Available columns: {list(df.columns)}"
        )

    # Add unique_id if not provided
    if id_column is None or id_column not in df.columns:
        df["unique_id"] = df.index
        id_column = "unique_id"
    else:
        # Rename the specified id column to unique_id for consistency
        if id_column != "unique_id":
            df = df.rename(columns={id_column: "unique_id"})

    return df


def load_dataset_from_hub(
    dataset_name: str, split: str, prompt_column: str, id_column: str = None
) -> pd.DataFrame:
    """Load dataset from HuggingFace Hub."""
    try:
        # Load dataset from HuggingFace Hub
        ds = datasets.load_dataset(dataset_name)[split]
        df = ds.to_pandas()

        # Validate prompt column exists
        if prompt_column not in df.columns:
            raise ValueError(
                f"Prompt column '{prompt_column}' not found in dataset. Available columns: {list(df.columns)}"
            )

        # Add unique_id if not provided
        if id_column is None or id_column not in df.columns:
            df["unique_id"] = df.index
            id_column = "unique_id"
        else:
            # Rename the specified id column to unique_id for consistency
            if id_column != "unique_id":
                df = df.rename(columns={id_column: "unique_id"})

        return df

    except Exception as e:
        raise ValueError(
            f"Failed to load dataset '{dataset_name}' from HuggingFace Hub: {e}"
        )


def load_dataset(
    dataset_source: str, prompt_column: str, id_column: str = None, split: str = "train"
) -> pd.DataFrame:
    """Load dataset from file or HuggingFace Hub."""
    # Check if it's a file path
    if "/" in dataset_source and (
        Path(dataset_source).exists() or "." in Path(dataset_source).suffix
    ):
        return load_dataset_from_file(dataset_source, prompt_column, id_column)
    else:
        # Assume it's a HuggingFace dataset name
        return load_dataset_from_hub(dataset_source, split, prompt_column, id_column)


def get_output_path(
    output_dir: str,
    dataset_name: str,
    model_name: str,
    algorithm: ScalingAlgorithm,
    budget: int,
) -> Path:
    """Generate hierarchical output path."""
    # Clean names for file system
    dataset_clean = Path(dataset_name).stem  # Remove extension
    model_clean = model_name.replace("/", "-")

    # All algorithms except self-consistency use reward models
    alg_dir = algorithm.value

    output_path = (
        Path(output_dir) / dataset_clean / model_clean / alg_dir / f"budget_{budget}"
    )
    return output_path


def init_algorithm(
    alg: ScalingAlgorithm,
    model_name: str,
    rm_name: str,
    rm_device: str,
    rm_agg_method: AggregationMethod,
    tokens_per_step: int = None,
):
    """Initialize scaling algorithm."""
    if alg == ScalingAlgorithm.SELF_CONSISTENCY:
        # No extraction function needed for pure generation
        return SelfConsistency(lambda x: x)  # Identity function
    elif alg == ScalingAlgorithm.BEAM_SEARCH:
        if tokens_per_step is None:
            raise ValueError("tokens_per_step is required for beam search algorithm")
        sg = StepGeneration(max_steps=50, tokens_per_step=tokens_per_step)
        prm = LocalVllmProcessRewardModel(
            model_name=rm_name, device=rm_device, aggregation_method=rm_agg_method
        )
        return BeamSearch(sg, prm, beam_width=4)
    elif alg == ScalingAlgorithm.PARTICLE_FILTERING:
        if tokens_per_step is None:
            raise ValueError(
                "tokens_per_step is required for particle filtering algorithm"
            )
        sg = StepGeneration(max_steps=50, tokens_per_step=tokens_per_step)
        prm = LocalVllmProcessRewardModel(
            model_name=rm_name, device=rm_device, aggregation_method=rm_agg_method
        )
        return ParticleFiltering(sg, prm)
    elif alg == ScalingAlgorithm.BEST_OF_N:
        # BestOfN can directly use LocalVllmProcessRewardModel
        prm = LocalVllmProcessRewardModel(
            model_name=rm_name, device=rm_device, aggregation_method=rm_agg_method
        )
        return BestOfN(reward_model=prm)
    else:
        raise ValueError(f"Unknown algorithm: {alg}")


def serialize_result_data(result_obj) -> Dict[str, Any]:
    """Convert result object to serializable dictionary."""
    result_dict = {}

    # Always capture the_one
    result_dict["the_one"] = result_obj.the_one

    # Capture algorithm-specific attributes
    if hasattr(result_obj, "responses"):
        result_dict["responses"] = result_obj.responses
    if hasattr(result_obj, "response_counts"):
        result_dict["response_counts"] = dict(result_obj.response_counts)
    if hasattr(result_obj, "scores"):
        result_dict["scores"] = result_obj.scores
    if hasattr(result_obj, "log_weights_lst"):
        result_dict["log_weights_lst"] = result_obj.log_weights_lst
    if hasattr(result_obj, "selected_index"):
        result_dict["selected_index"] = result_obj.selected_index
    if hasattr(result_obj, "steps_used"):
        result_dict["steps_used"] = result_obj.steps_used
    if hasattr(result_obj, "steps_used_lst"):
        result_dict["steps_used_lst"] = result_obj.steps_used_lst

    return result_dict


def check_existing_result(output_path: Path, unique_id: str) -> bool:
    """Check if result already exists for this unique_id."""
    result_file = output_path / "results.jsonl"
    if not result_file.exists():
        return False

    # Check if this unique_id is already processed
    try:
        df_existing = pd.read_json(result_file, lines=True)
        return unique_id in df_existing["unique_id"].values
    except:
        return False


@click.command()
@click.option(
    "--dataset_source",
    type=str,
    required=True,
    help="Dataset source: local file path (CSV/JSON/JSONL) or HuggingFace dataset name",
)
@click.option(
    "--split",
    type=str,
    default="train",
    help="Dataset split to use (for HuggingFace datasets)",
)
@click.option(
    "--prompt_column", type=str, default="prompt", help="Column name containing prompts"
)
@click.option(
    "--id_column",
    type=str,
    default=None,
    help="Column name for unique IDs (auto-generated if not provided)",
)
@click.option(
    "--model_name",
    type=str,
    required=True,
    help="Model to use for inference-time scaling",
)
@click.option("--is_async", is_flag=True, default=True, help="Use async mode")
@click.option("--max_tokens", type=int, default=None, help="Max tokens for generation")
@click.option(
    "--temperature", type=float, default=None, help="Temperature for generation"
)
@click.option(
    "--max_concurrency", type=int, default=8, help="Max concurrency for async mode"
)
@click.option("--endpoint", type=str, help="API endpoint")
@click.option("--api_key", type=str, default="NO_API_KEY", help="API key")
@click.option(
    "--rm_name", type=str, default="Qwen/Qwen2.5-Math-PRM-7B", help="Reward model name"
)
@click.option("--rm_device", type=str, default="cpu", help="Reward model device")
@click.option(
    "--rm_agg_method",
    type=click.Choice([e.value for e in AggregationMethod]),
    default="model",
    callback=lambda ctx, param, value: AggregationMethod(value),
    help="Reward model aggregation method",
)
@click.option(
    "--alg",
    type=click.Choice([e.value for e in ScalingAlgorithm]),
    required=True,
    callback=lambda ctx, param, value: ScalingAlgorithm(value),
    help="Scaling algorithm to use",
)
@click.option(
    "--subset",
    type=str,
    default=None,
    help="Subset of dataset (e.g., ':10', '5:', '5:10')",
)
@click.option(
    "--budgets",
    type=str,
    default="1,2,4,8",
    callback=lambda ctx, param, value: [int(b) for b in value.split(",")],
    help="Comma-separated list of budgets",
)
@click.option(
    "--output_dir", type=str, default="generation_results", help="Base output directory"
)
@click.option(
    "--force_run", is_flag=True, default=False, help="Force re-run existing results"
)
@click.option(
    "--tokens_per_step",
    type=int,
    default=None,
    help="Tokens per step for StepGeneration",
)
def main(
    dataset_source: str,
    split: str,
    prompt_column: str,
    id_column: str,
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
    force_run: bool,
    tokens_per_step: int,
):
    """Generate responses using inference-time scaling algorithms."""

    # Display configuration in a nice table
    ctx = click.get_current_context()

    console.print("\n[bold blue]🚀 Generation Script Started[/bold blue]")

    config_table = Table(title="Configuration", box=box.ROUNDED)
    config_table.add_column("Parameter", style="cyan", no_wrap=True)
    config_table.add_column("Value", style="green")

    for param_name, param_value in ctx.params.items():
        config_table.add_row(param_name, str(param_value))

    console.print(config_table)

    # Load dataset
    with console.status("[bold green]Loading dataset...", spinner="dots"):
        df = load_dataset(dataset_source, prompt_column, id_column, split)

    # Determine dataset name for output organization
    if "/" in dataset_source and Path(dataset_source).exists():
        dataset_name = Path(dataset_source).stem
    else:
        # HuggingFace dataset - use the dataset name
        dataset_name = dataset_source.replace("/", "-")

    console.print(
        f"[green]✓[/green] Loaded {len(df)} examples from dataset '[cyan]{dataset_name}[/cyan]' (split: {split})"
    )

    # Apply subset if specified
    if subset is not None:
        try:
            if ":" in subset:
                parts = subset.split(":")
                if len(parts) == 2:
                    start = int(parts[0]) if parts[0] else None
                    end = int(parts[1]) if parts[1] else None
                    df = df.iloc[start:end]
            else:
                df = df.iloc[[int(subset)]]
            console.print(f"[yellow]📝[/yellow] Using subset: {len(df)} examples")
        except ValueError:
            console.print(
                f"[red]❌[/red] Invalid subset format: {subset}, using full dataset"
            )

    # Initialize language model
    with console.status("[bold green]Initializing language model...", spinner="dots"):
        if endpoint is not None:
            lm = OpenAICompatibleLanguageModel(
                endpoint=endpoint,
                api_key=api_key,
                model_name=model_name,
                system_prompt=QWEN_SYSTEM_PROMPT
                if "qwen" in model_name.lower()
                else SAL_STEP_BY_STEP_SYSTEM_PROMPT,
                is_async=is_async,
                temperature=temperature,
                max_tokens=max_tokens,
                max_concurrency=max_concurrency,
            )
        else:
            raise ValueError("Endpoint must be provided")

    console.print(
        f"[green]✓[/green] Language model initialized: [cyan]{model_name}[/cyan]"
    )

    # Initialize algorithm
    with console.status(
        f"[bold green]Initializing {alg.value} algorithm...", spinner="dots"
    ):
        scaling_alg = init_algorithm(
            alg, model_name, rm_name, rm_device, rm_agg_method, tokens_per_step
        )

    console.print(f"[green]✓[/green] Algorithm initialized: [cyan]{alg.value}[/cyan]")

    # Process each budget
    for budget in budgets:
        console.print(f"\n[bold magenta]{'🎯 ' * 20}[/bold magenta]")
        console.print(
            Panel(
                f"[bold white]Processing Budget: {budget}[/bold white]",
                style="magenta",
                box=box.DOUBLE,
            )
        )

        # Get output path for this configuration
        output_path = get_output_path(output_dir, dataset_name, model_name, alg, budget)
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / "results.jsonl"

        console.print(f"[blue]📁[/blue] Output path: [cyan]{output_path}[/cyan]")

        # Track progress
        total_rows = len(df)
        processed_count = 0
        skipped_count = 0

        # Process each row with progress tracking
        results = []

        # Create rich progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"[cyan]Processing Budget {budget}", total=total_rows
            )

            for idx, row in df.iterrows():
                unique_id = row["unique_id"]
                prompt = row[prompt_column]

                progress.update(
                    task,
                    description=f"[cyan]Budget {budget} | Row {unique_id} | Processed: {processed_count} | Skipped: {skipped_count}",
                )

                # Check if already processed (unless force_run)
                if not force_run and check_existing_result(output_path, unique_id):
                    skipped_count += 1
                    progress.advance(task)
                    continue

                try:
                    # Run inference with full result capture
                    result_obj = scaling_alg.infer(
                        lm, prompt, budget, return_response_only=False
                    )

                    # Create result record
                    result_record = {
                        "unique_id": unique_id,
                        "budget": budget,
                        "algorithm": alg.value,
                        "model": model_name,
                        "prompt": prompt,
                    }

                    # Add original dataset columns
                    for col in df.columns:
                        if col not in ["unique_id", prompt_column]:
                            result_record[f"{col}"] = row[col]

                    # Add generation results
                    generation_data = serialize_result_data(result_obj)
                    for key, value in generation_data.items():
                        result_record[f"generation_{key}"] = value

                    # Add metadata
                    result_record["generation_metadata"] = {
                        "model": model_name,
                        "algorithm": alg.value,
                        "budget": budget,
                        "rm_name": rm_name
                        if alg != ScalingAlgorithm.SELF_CONSISTENCY
                        else None,
                        "rm_agg_method": rm_agg_method.value
                        if alg != ScalingAlgorithm.SELF_CONSISTENCY
                        else None,
                        "tokens_per_step": tokens_per_step,
                    }

                    results.append(result_record)
                    processed_count += 1

                except KeyboardInterrupt:
                    console.print(
                        f"\n[yellow]⚠️[/yellow] Keyboard interrupt detected. Saving {len(results)} results..."
                    )
                    break
                except Exception as e:
                    console.print(
                        f"[red]❌[/red] Error processing row {unique_id}: {e}"
                    )
                    continue
                finally:
                    progress.advance(task)

        # Save results for this budget
        if results:
            console.print(
                f"\n[blue]💾[/blue] Saving {len(results)} results to [cyan]{result_file}[/cyan]"
            )

            # Load existing results if any
            existing_results = []
            if result_file.exists() and not force_run:
                try:
                    existing_df = pd.read_json(result_file, lines=True)
                    existing_results = existing_df.to_dict("records")
                except:
                    pass

            # Combine and deduplicate
            all_results = existing_results + results
            df_results = pd.DataFrame(all_results)

            if not df_results.empty:
                # Remove duplicates keeping the latest
                df_results = df_results.drop_duplicates(
                    subset=["unique_id"], keep="last"
                )

                # Save to JSONL
                df_results.to_json(result_file, orient="records", lines=True)
                console.print(f"[green]✓[/green] Saved {len(df_results)} total results")

        # Budget completion summary
        summary_table = Table(box=box.SIMPLE)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Count", style="green")
        summary_table.add_row("Processed", str(processed_count))
        summary_table.add_row("Skipped", str(skipped_count))
        summary_table.add_row("Total", str(processed_count + skipped_count))

        console.print(f"\n[green]✅[/green] Budget {budget} completed:")
        console.print(summary_table)

    # Final completion message
    console.print("\n" + "🎉" * 20)
    console.print(
        Panel(
            f"[bold green]Generation Script Completed Successfully![/bold green]\n"
            f"[white]Results saved in hierarchical structure under:[/white]\n"
            f"[cyan]{output_dir}[/cyan]",
            style="green",
            box=box.DOUBLE,
        )
    )


if __name__ == "__main__":
    main()
