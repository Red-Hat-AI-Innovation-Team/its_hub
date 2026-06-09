# Chapter 1 — The Problem: Inference-Time Scaling

> *Previous: [README](README.md) · Next: [Architecture](02-architecture.md)*

## Why run a model more than once?

A language model that has finished training is a fixed function: given a prompt, it samples a
continuation. For an easy question, one sample is plenty. For a hard, multi-step math problem, a single
greedy pass is a gamble — the model might take a wrong turn at step 3 and never recover, even though it
*could* have solved the problem on a different roll of the dice.

There are two ways to get better answers:

1. **Train a bigger/better model** (expensive, slow, and you may not control the weights).
2. **Spend more compute at *inference* time** on the model you already have.

`its_hub` is entirely about path #2. This is called **inference-time scaling** (ITS), also known as
**test-time compute**. The core empirical finding — established across several research lines — is that
*for reasoning tasks, trading extra inference compute for accuracy is often a better deal than the same
compute spent on training*. A small model that is allowed to "think" with a clever search procedure can
match or beat a much larger model answering in one shot.

The library frames this as a single question:

> **Given a fixed LM and a compute `budget`, how do we spend that budget to maximize answer quality?**

Every algorithm in `its_hub` is a different answer to that question.

## Two axes that organize everything

The five algorithms differ along two axes. Holding this 2×2 in your head makes the rest of the story
click.

```
                    │ scores WHOLE answers          │ scores PARTIAL answers (per step)
  ──────────────────┼───────────────────────────────┼──────────────────────────────────
   keep ALL,         │  Self-Consistency             │   (—)
   pick by agreement │  (vote on final answers)      │
  ──────────────────┼───────────────────────────────┼──────────────────────────────────
   keep the BEST,    │  Best-of-N                    │   Beam Search        (deterministic)
   pick by a reward  │  (rerank with an ORM)         │   Particle Filtering (probabilistic)
                     │                               │   Entropic PF        (probabilistic+anneal)
```

- **Axis 1 — *when* do you judge?** Either you generate complete answers and judge them at the end
  (**outcome** supervision), or you judge the reasoning *as it unfolds*, one step at a time
  (**process** supervision). Process supervision is more powerful because it can prune a bad path
  *before* wasting compute finishing it — but it requires a step-level reward model.

- **Axis 2 — *how* do you choose?** Either by **consensus** (the most common answer wins — no reward
  model needed), or by a **reward** (a model assigns a score and you keep the high scorers).

The four reward-based cells are further split by *how* they keep the good candidates: **deterministically**
(sort and keep the top-k — Beam Search) or **probabilistically** (resample in proportion to weights —
Particle Filtering). That deterministic-vs-probabilistic distinction is the heart of
[Chapter 6](06-beam-search.md) vs [Chapter 7](07-particle-filtering.md).

## The four (well, five) methodologies in one breath

| Methodology | The intuition | Reward signal | Chapter |
|-------------|---------------|---------------|---------|
| **Self-Consistency** | "Ask 10 times; the answer most of them agree on is probably right." | none — majority vote | [05](05-self-consistency-and-best-of-n.md) |
| **Best-of-N** | "Generate 10 full answers; a judge picks the best one." | **outcome** (ORM) | [05](05-self-consistency-and-best-of-n.md) |
| **Beam Search** | "Grow several reasoning paths step by step; at each step keep the top few." | **process** (PRM) | [06](06-beam-search.md) |
| **Particle Filtering** | "Grow many paths; at each step, *resample* — clone the promising ones, drop the weak ones." | **process** (PRM) | [07](07-particle-filtering.md) |
| **Entropic PF** | "Same, but don't let the swarm collapse onto one lucky path too early." | **process** (PRM) | [08](08-entropic-particle-filtering.md) |

There is also a meta-algorithm, the **Planning Wrapper** ([Chapter 10](10-planning-wrapper.md)), that
first asks the model to brainstorm a few *approaches*, then runs any of the above once per approach.

## Where this library sits in the research landscape

These ideas are not invented here; `its_hub` is a clean, production-minded *implementation* of several
research threads:

- **Self-Consistency** comes from Wang et al., *"Self-Consistency Improves Chain of Thought Reasoning
  in Language Models"* (2022).
- **Best-of-N / verifier reranking** traces to Cobbe et al.'s GSM8K *verifiers* (2021) and the broader
  "train a reward model, rerank samples" recipe.
- **Process supervision** — judging reasoning step by step — is the subject of Lightman et al., *"Let's
  Verify Step by Step"* (2023), which showed step-level reward models (PRMs) outperform outcome-only
  ones on math.
- **Particle-based Monte Carlo for inference scaling** — the probabilistic core of this repo — follows
  Puri et al., *"A Probabilistic Inference Approach to Inference-Time Scaling of LLMs using
  Particle-Based Monte Carlo Methods"* (2025), which recasts ITS as **Sequential Monte Carlo** sampling
  from a reward-tilted distribution.
- **Entropic annealing** borrows the *tempering* idea from the classical SMC / annealed-importance-
  sampling literature (Neal 2001; Del Moral, Doucet & Jasra 2006) to fight particle degeneracy.

Full, verified citations are in [Chapter 99](99-glossary-and-references.md).

## What the library deliberately is *not*

A recurring design decision — visible throughout the code — is **minimalism at the core**. The base
install depends only on `numpy` and `typing-extensions`. Everything heavier (an OpenAI client, vLLM,
reward models, datasets) is an *optional extra*. This is why a gateway team can adopt the algorithm
interfaces with almost no dependency footprint, and why particle filtering — which needs a GPU-hosted
reward model — is quarantined behind the `[experimental]` extra. We return to this in
[Chapter 2](02-architecture.md) and [Chapter 12](12-running-it.md).

---

*Next: [Chapter 2 — The Architecture](02-architecture.md), where we meet the contracts every algorithm
shares.*
