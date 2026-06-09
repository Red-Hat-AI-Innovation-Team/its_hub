# Inside `its_hub` — A Deep Dive into Entropic Particle Filtering for LLMs

> A single, connected story that takes you from *"why would you run a model more than once?"*
> all the way down to *"which exact line of code turns a reward into a particle weight, and is it
> adding or multiplying?"*

This `documentation/` folder is **not** a replacement for the user-facing [`docs/`](../docs/) site
(installation, quick-start, API reference). That site tells you *how to use* the library. This one
tells you *how it actually works* — the mechanics, the math, the design decisions, and the precise
place in the source where each idea lives. Read it front to back like a book, or jump to the chapter
you need.

The repository is named **`entropic-particle-filter`**, and the published library is **`its_hub`**
("**I**nference-**T**ime **S**caling hub"). The name is a promise: the crown-jewel algorithm here is
**Entropic Particle Filtering (ePF)**, and a large part of this story builds toward understanding it.

---

## The one-paragraph answer (for the impatient)

`its_hub` makes a *fixed* language model give *better* answers by spending more compute **at inference
time** instead of retraining. It offers five algorithms on one interface. Four are about *trajectories*:
generate several candidate solutions, **score** them with a reward model, and **keep the good ones**.
The probabilistic crown jewel — particle filtering — treats each candidate as a "particle" in a
Sequential Monte Carlo sampler: at every reasoning step it scores the partial solution with a **Process
Reward Model (PRM)**, turns that score (a probability in $[0,1]$) into a **log-weight** via the *logit*
function $\operatorname{logit}(s)=\ln\frac{s}{1-s}$, and **resamples** particles in proportion to those
weights so compute flows toward promising reasoning paths. **Entropic** Particle Filtering adds a
*temperature* that flattens the resampling distribution early on — when the particles would otherwise
collapse onto one lucky guess — preserving diversity for hard, multi-step problems.

> **The headline "multiply vs. add" answer** (the question this deep-dive was commissioned to settle):
> Per particle, the weight at step *t* is the **logit of the PRM's score of the whole partial trajectory
> so far** — it is *re-derived* each step, not accumulated by a running sum or product in the filter.
> Across particles, weights are combined with a **softmax** (so a particle's resample probability is
> $\propto s/(1-s)$, the *odds*). Any *multiplication of per-step rewards* happens **inside** the PRM's
> `aggregation_method="prod"`, not in the particle filter. See
> [Chapter 7](07-particle-filtering.md) for the full derivation.

---

## How to read this

| # | Chapter | What you'll learn |
|---|---------|-------------------|
| — | [README](README.md) (this file) | The map, the budget cheat-sheet, the headline answer |
| 01 | [The Problem: Inference-Time Scaling](01-the-problem.md) | *Why* scale at inference; the four methodologies at a glance |
| 02 | [The Architecture](02-architecture.md) | `api/` vs `core/`, the universal `ainfer`/`budget`/`the_one` contract |
| 03 | [Generating Text](03-generating-text.md) | LMs, the orchestrator, step-by-step generation — and why **no logprobs flow** |
| 04 | [Reward Models](04-reward-models.md) | PRM vs ORM, `LLMJudge`, `LocalVllmProcessRewardModel`, what a "reward" *is* |
| 05 | [The Simple Scalers](05-self-consistency-and-best-of-n.md) | Self-Consistency (voting) and Best-of-N (rerank) |
| 06 | [Beam Search](06-beam-search.md) | Step-by-step *deterministic* search — the foil for particle filtering |
| 07 | [Particle Filtering](07-particle-filtering.md) | **The probabilistic core + the full log-weight derivation** |
| 08 | [Entropic Particle Filtering](08-entropic-particle-filtering.md) | **The namesake**: degeneracy, ESS, entropy, temperature annealing |
| 09 | [Particle Gibbs & the Family](09-particle-gibbs.md) | One class, four algorithms: PF, ePF, PG, PGAS |
| 10 | [The Planning Wrapper](10-planning-wrapper.md) | A meta-algorithm that wraps any of the above |
| 11 | [Putting It Together](11-putting-it-together.md) | End-to-end data flow; choosing an algorithm |
| 12 | [Running It](12-running-it.md) | The conda env, the tests, the inference paths (incl. this machine's gotchas) |
| 99 | [Glossary & References](99-glossary-and-references.md) | Every term + the verified bibliography |

Runnable, GPU-free demonstrations live in [`snippets/`](snippets/) and are referenced from the chapters.

---

## The budget cheat-sheet

Every algorithm takes the same `budget: int`, but each *spends* it differently. Keep this table handy.

| Algorithm | Class | What `budget` means | Reward model | Deterministic? |
|-----------|-------|---------------------|--------------|----------------|
| Self-Consistency | `SelfConsistency` | # parallel generations to vote over | none (voting) | no (random tie-break) |
| Best-of-N | `BestOfN` | # parallel generations to rerank | **ORM** (outcome) | yes (argmax) |
| Beam Search | `BeamSearch` | total generations = `num_beams × beam_width` | **PRM** (process) | yes (argmax) |
| Particle Filtering | `ParticleFiltering` | # particles maintained | **PRM** (process) | no (resampling) |
| Entropic Particle Filtering | `EntropicParticleFiltering` | # particles maintained | **PRM** (process) | no (resampling + annealing) |
| *(Particle Gibbs)* | `ParticleGibbs` | `num_particles × num_iterations` | **PRM** (process) | configurable |
| *(Planning Wrapper)* | `PlanningWrapper` | 1 (plan) + remainder split over approaches | inherits base | inherits base |

Two more universal facts you'll see everywhere:

- **`ainfer(...)` is the async primary; `infer(...)` is a thin `asyncio.run` wrapper** around it
  ([`its_hub/api/algorithm.py:64-94`](../its_hub/api/algorithm.py#L64-L94)).
- **Every result object exposes `.the_one`** — the single chosen response dict
  ([`its_hub/api/algorithm.py:8-25`](../its_hub/api/algorithm.py#L8-L25)). Call with
  `return_response_only=False` to get the whole result object (all candidates, all scores/weights).

---

## A note on accuracy

Every code reference in these chapters points at a real `path:line` in this repository and was checked
against the source while writing. Short snippets are quoted *verbatim* (especially the weight math) so
you can trust them without re-deriving. Citations to outside research were verified before inclusion;
they live in [Chapter 99](99-glossary-and-references.md).
