# Inside `its_hub` — A Deep Dive into Entropic Particle Filtering for LLMs

> A single, connected story that takes you from *"why would you run a model more than once?"*
> all the way down to *"which exact line of code turns the model's own confidence into a particle
> weight, and is it adding or multiplying?"*

This `documentation/` folder tells you *how the library actually works* — the mechanics, the math, the
design decisions, and the precise place in the source where each idea lives. Read it front to back
like a book, or jump to the chapter you need.

The repository is named **`entropic-particle-filter`**, and the published library is **`its_hub`**
("**I**nference-**T**ime **S**caling hub"). The name is a promise: the crown-jewel algorithm here is
**Entropic Particle Filtering (ePF)**, and a large part of this story builds toward understanding it.

---

## The one-paragraph answer (for the impatient)

`its_hub` makes a *fixed* language model give *better* answers by spending more compute **at inference
time** instead of retraining. It offers two algorithms on one interface, both probabilistic: each
candidate solution is a "particle" in a Sequential Monte Carlo sampler. At every reasoning step the
generator's **own token logprobs** are summarized into a *self-certainty* score — a confidence
$s=e^{\bar\ell}\in(0,1]$, where $\bar\ell$ is the step's mean token logprob — turned into a
**log-weight** via the *logit* function $\operatorname{logit}(s)=\ln\frac{s}{1-s}$, and the filter
**resamples** particles in proportion to those weights so compute flows toward confident reasoning
paths. No external reward model is involved anywhere: the model judges itself. **Entropic** Particle
Filtering adds a *temperature* that flattens the resampling distribution early on — when the particles
would otherwise collapse onto one lucky guess — preserving diversity for hard, multi-step problems.

> **The headline "multiply vs. add" answer** (the question this deep-dive was commissioned to settle):
> Per particle, the weight at step *t* is the **logit of that step's self-certainty** —
> $\operatorname{logit}(e^{\bar\ell_t})$, computed by `_self_certainty_logweight` via `_inv_sigmoid`
> in [`particle_filtering.py`](../its_hub/core/algorithms/particle_filtering.py). It is *re-derived*
> from each step's own logprobs, not accumulated by a running sum or product in the filter. Across
> particles, the most recent weights are combined with a **softmax** at resampling (so a particle's
> resample probability is $\propto s/(1-s)$, the *odds* of its step confidence). Nothing multiplies
> per-step scores anywhere — the PRM whose `aggregation_method="prod"` once did has been removed from
> the library. See [Chapter 7](07-particle-filtering.md) for the full derivation.

---

## How to read this

| # | Chapter | What you'll learn |
|---|---------|-------------------|
| — | [README](README.md) (this file) | The map, the budget cheat-sheet, the headline answer |
| 01 | [The Problem: Inference-Time Scaling](01-the-problem.md) | *Why* scale at inference; the methodology landscape at a glance |
| 02 | [The Architecture](02-architecture.md) | `api/` vs `core/`, the universal `ainfer`/`budget`/`the_one` contract |
| 03 | [Generating Text](03-generating-text.md) | LMs, the orchestrator, step-by-step generation — and how token **logprobs flow back** |
| 07 | [Particle Filtering](07-particle-filtering.md) | **The probabilistic core + the full log-weight derivation** |
| 08 | [Entropic Particle Filtering](08-entropic-particle-filtering.md) | **The namesake**: degeneracy, ESS, entropy, temperature annealing |
| 11 | [Putting It Together](11-putting-it-together.md) | End-to-end data flow; choosing between PF and ePF |
| 12 | [Running It](12-running-it.md) | The conda env, the tests, the inference paths (incl. this machine's gotchas) |
| 99 | [Glossary & References](99-glossary-and-references.md) | Every term + the verified bibliography |
| — | [Audio / MMAU changes](audio-mmau-changes.md) | How structured (audio) messages ride through the step path |

(Chapters 04–06 and 09–10 covered the reward models and the Self-Consistency / Best-of-N / Beam Search /
Particle Gibbs / Planning-Wrapper algorithms; those were removed from the library and their chapters
retired — hence the numbering gaps.)

Runnable, GPU-free demonstrations live in [`snippets/`](snippets/) and are referenced from the chapters.

---

## The budget cheat-sheet

Both algorithms take the same `budget: int`, and both spend it the same way: **`budget` is the number
of particles maintained**. Keep this table handy.

| Algorithm | Class | What `budget` means | Weight source | Deterministic? |
|-----------|-------|---------------------|---------------|----------------|
| Particle Filtering | `ParticleFiltering` | # particles maintained | **self-certainty** (generator's own logprobs) | no (resampling) |
| Entropic Particle Filtering | `EntropicParticleFiltering` | # particles maintained | **self-certainty** (generator's own logprobs) | no (resampling + annealing) |

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
