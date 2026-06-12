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

`its_hub` gives one answer to that question — **Particle Filtering**, plus its annealed variant
**Entropic Particle Filtering**. (Earlier versions of the library also shipped Self-Consistency,
Best-of-N, and Beam Search; those have been removed, and the codebase is now PF/EPF-only.)

## Where PF sits in the design space

The broader ITS design space is organized by two axes; PF occupies a specific corner of it, and knowing
the alternatives makes its choices click.

- **Axis 1 — *when* do you judge?** Either you generate complete answers and judge them at the end
  (**outcome** supervision), or you judge the reasoning *as it unfolds*, one step at a time
  (**process** supervision). Process supervision is more powerful because it can prune a bad path
  *before* wasting compute finishing it. PF judges per step.

- **Axis 2 — *how* do you keep the good candidates?** **Deterministically** (sort and keep the top-k —
  beam search) or **probabilistically** (resample in proportion to weights — particle filtering). PF
  resamples.

So PF is the *process-supervised, probabilistic* corner: grow many reasoning paths, and at each step
*resample* — clone the promising ones, drop the weak ones. One twist relative to the classic recipe:
the per-step judging signal here is **not** a separate reward model. It is the generator's own
token log-probabilities — *self-certainty* — which is the whole story of
[Chapter 7](07-particle-filtering.md).

## The two methodologies in one breath

| Methodology | The intuition | Weight signal | Chapter |
|-------------|---------------|---------------|---------|
| **Particle Filtering** | "Grow many paths; at each step, *resample* — clone the promising ones, drop the weak ones." | **self-certainty** (the generator's own token logprobs) | [07](07-particle-filtering.md) |
| **Entropic PF** | "Same, but don't let the swarm collapse onto one lucky path too early." | **self-certainty** + temperature annealing | [08](08-entropic-particle-filtering.md) |

## Where this library sits in the research landscape

These ideas are not invented here; `its_hub` is a clean, production-minded *implementation* of several
research threads:

- **Process supervision** — judging reasoning step by step rather than only at the end — is the subject
  of Lightman et al., *"Let's Verify Step by Step"* (2023). (That paper does it with a trained
  step-level reward model; this repo replaces the PRM with the generator's own self-certainty.)
- **Particle-based Monte Carlo for inference scaling** — the probabilistic core of this repo — follows
  Puri et al., *"A Probabilistic Inference Approach to Inference-Time Scaling of LLMs using
  Particle-Based Monte Carlo Methods"* (2025), which recasts ITS as **Sequential Monte Carlo** sampling
  from a reward-tilted distribution.
- **Entropic annealing** borrows the *tempering* idea from the classical SMC / annealed-importance-
  sampling literature (Neal 2001; Del Moral, Doucet & Jasra 2006) to fight particle degeneracy.

Full, verified citations are in [Chapter 99](99-glossary-and-references.md).

## What the library deliberately is *not*

A recurring design decision — visible throughout the code — is **minimalism at the core**. The base
install depends only on `numpy` and `typing-extensions`. Everything heavier is an *optional extra*: the
OpenAI-compatible client lives behind `[lm]`, and the MMAU-Pro benchmark dependencies behind
`[benchmark]`. There is no reward model anywhere — the particle weights come from the generator's own
token logprobs — so no GPU-hosted judge is needed. This is why a gateway team can adopt the algorithm
interfaces with almost no dependency footprint. We return to this in
[Chapter 2](02-architecture.md) and [Chapter 12](12-running-it.md).

---

*Next: [Chapter 2 — The Architecture](02-architecture.md), where we meet the contracts every algorithm
shares.*
