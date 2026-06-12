# Chapter 99 — Glossary & References

> *Previous: [Running It](12-running-it.md) · Back to [README](README.md)*

## Glossary

**Inference-time scaling (ITS) / test-time compute.** Spending extra compute *at inference* (multiple
samples, search) to improve answer quality from a *fixed* model, rather than retraining. The premise of
the whole library ([Chapter 1](01-the-problem.md)).

**`budget`.** The integer compute allowance passed to every algorithm; for PF and ePF it is the number
of particles maintained (see the [cheat-sheet](README.md#the-budget-cheat-sheet)).

**`the_one`.** The single chosen response dict returned by every result object
([`api/algorithm.py:8-25`](../its_hub/api/algorithm.py#L8-L25)).

**`ainfer` / `infer`.** The async primary entry point and its synchronous `asyncio.run` wrapper
([Chapter 2](02-architecture.md)).

**Reward models (PRM / ORM) — removed.** Earlier versions weighted particles with an external Process
Reward Model (and offered ORM-based reranking). All reward models were removed from the library; the
generator's own self-certainty is now the only weight source.

**Self-certainty.** The generator's own confidence in a step, derived from its token logprobs: either
the mean token logprob (default, `self_certainty_signal="mean_logprob"`) or the negative mean per-token
entropy over the top-k alternatives (`"entropy"`). Summarized per step by `summarize_step_logprobs`
([`core/utils.py`](../its_hub/core/utils.py)) and turned into a log-weight by
`_self_certainty_logweight` ([`particle_filtering.py`](../its_hub/core/algorithms/particle_filtering.py)).

**Particle.** One candidate reasoning trajectory in the particle filter — its steps, a stopped flag, and
a history of log-weights (the `Particle` dataclass in
[`particle_filtering.py`](../its_hub/core/algorithms/particle_filtering.py)).

**Log-weight.** A particle's weight in log-space. With the default `self_certainty_style="logit"` it is
the **logit** of the step's self-certainty: $w=\operatorname{logit}(s)=\ln\frac{s}{1-s}$ with
$s=e^{\bar\ell}$ (mean step logprob), computed by `_inv_sigmoid` inside `_self_certainty_logweight`
([`particle_filtering.py`](../its_hub/core/algorithms/particle_filtering.py)); style `"raw"` uses the
confidence directly. Re-derived each step, not accumulated. See
[Chapter 7](07-particle-filtering.md) for the full derivation.

**Logit / inverse sigmoid.** $\operatorname{logit}(s)=\ln\frac{s}{1-s}$, the inverse of the sigmoid.
Maps a probability $[0,1]$ to a log-odds on the whole real line.

**Sequential Monte Carlo (SMC) / particle filter.** A family of algorithms that approximate a target
distribution with a weighted set of samples, evolved by *sample → weight → resample*. The probabilistic
core of this library (Gordon et al. 1993; Doucet et al. 2001).

**Importance sampling.** Estimating properties of a target distribution using samples from an easier
proposal, corrected by weights. Resampling weights in the particle filter are importance weights.

**Resampling.** Drawing a new particle population with replacement in proportion to weights — clones
strong particles, drops weak ones. `multinomial` (i.i.d.) or `systematic` (low-variance comb) —
`_resampling_multinomial` / `_resampling_systematic` in
[`particle_filtering.py`](../its_hub/core/algorithms/particle_filtering.py).

**Effective Sample Size (ESS).** $\mathrm{ESS}=1/\sum_i p_i^2 \in [1,N]$. A diagnostic for weight
concentration; low ESS signals degeneracy (`_effective_sample_size` in
[`particle_filtering.py`](../its_hub/core/algorithms/particle_filtering.py)).

**Particle degeneracy.** The pathology where weight concentrates on one particle (ESS → 1). **Sample
impoverishment** is the follow-on loss of diversity after resampling clones it. The problem ePF fights
([Chapter 8](08-entropic-particle-filtering.md)).

**Normalized entropy.** $H_n = -\sum_i p_i\ln p_i / \ln N \in [0,1]$; another spread diagnostic, used by
ePF's entropy temperature schedule (`_entropy_n` in
[`particle_filtering.py`](../its_hub/core/algorithms/particle_filtering.py)).

**Entropic annealing / tempering.** Raising the resampling **temperature** $T>1$ to flatten the
distribution (preserve diversity) early, annealing back to $T=1$ later. Tempered softmax
$p_i(T)=\frac{e^{w_i/T}}{\sum_j e^{w_j/T}}$; implemented by ePF overriding `_weights_to_probabilities`
([Chapter 8](08-entropic-particle-filtering.md)).

**Orchestrator.** The component that fans out parallel LM calls with a concurrency cap
(`LMOrchestrator` uses `asyncio.TaskGroup` + a thread-safe semaphore). Still part of the public API,
though the particle filter drives generation through `StepGeneration` directly
([Chapter 3](03-generating-text.md)).

**StepGeneration.** The adapter that turns the LM into a one-step-at-a-time generator for the particle
filter, stopping on `max_steps` or a `stop_token`, and (with `return_logprobs=True`, which the filter
always requests) returning a per-step logprob summary alongside each step
([`lms/step_generation.py`](../its_hub/core/lms/step_generation.py)).

**PF / ePF.** Particle Filtering and Entropic Particle Filtering — the two algorithms in the library
(`ParticleFiltering` and its subclass `EntropicParticleFiltering`). The former `ParticleGibbs` family
(PG, PGAS, multi-iteration sweeps, reference particles) was removed.

## References

### The paper this repository implements

- **Puri, I., Sudalairaj, S., Xu, G., Xu, K., & Srivastava, A. (2025).** *A Probabilistic Inference
  Approach to Inference-Time Scaling of LLMs using Particle-Based Monte Carlo Methods* (a.k.a. *"Rollout
  Roulette"*). arXiv:2502.01618. NeurIPS 2025.
  <https://arxiv.org/abs/2502.01618> · project page: <https://probabilistic-inference-scaling.github.io/>
  > **Direct lineage:** co-author **Kai Xu** is the author of `its_hub` (see
  > [`pyproject.toml`](../pyproject.toml)). This library began as the reference implementation of that
  > paper; the particle-filtering machinery in [Chapter 7](07-particle-filtering.md) descends from the
  > paper's method, though the library has since replaced the paper's PRM weighting with the
  > generator's own self-certainty.

### Inference-time scaling & verification

- **Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., et al. (2022).** *Self-Consistency Improves Chain
  of Thought Reasoning in Language Models.* arXiv:2203.11171. — part of the methodology landscape
  surveyed in [Chapter 1](01-the-problem.md). <https://arxiv.org/abs/2203.11171>
- **Cobbe, K., Kosaraju, V., et al. (2021).** *Training Verifiers to Solve Math Word Problems* (GSM8K).
  arXiv:2110.14168. — the "generate many, rerank with a verifier" recipe behind Best-of-N.
  <https://arxiv.org/abs/2110.14168>
- **Lightman, H., Kosaraju, V., Burda, Y., et al. (2023).** *Let's Verify Step by Step.* arXiv:2305.20050.
  — process supervision / PRMs ([Chapter 1](01-the-problem.md)). <https://arxiv.org/abs/2305.20050>
- **Snell, C., Lee, J., Xu, K., & Kumar, A. (2024).** *Scaling LLM Test-Time Compute Optimally can be More
  Effective than Scaling Model Parameters.* arXiv:2408.03314. — the test-time-compute thesis.
  <https://arxiv.org/abs/2408.03314>

### Sequential Monte Carlo foundations

- **Gordon, N. J., Salmond, D. J., & Smith, A. F. M. (1993).** *Novel approach to nonlinear/non-Gaussian
  Bayesian state estimation.* IEE Proceedings F, 140(2), 107–113. — the bootstrap particle filter
  (sample → weight → resample).
- **Doucet, A., de Freitas, N., & Gordon, N. (eds.) (2001).** *Sequential Monte Carlo Methods in
  Practice.* Springer. — the standard SMC reference.
- **Neal, R. M. (2001).** *Annealed Importance Sampling.* Statistics and Computing, 11(2), 125–139. —
  tempering, the idea behind entropic annealing ([Chapter 8](08-entropic-particle-filtering.md)).
- **Del Moral, P., Doucet, A., & Jasra, A. (2006).** *Sequential Monte Carlo Samplers.* J. R. Statist.
  Soc. B, 68(3), 411–436. — adaptive tempering and ESS-based resampling.

### In-repo documents

- [`CLAUDE.md`](../CLAUDE.md) — developer commands and conventions.
- [`snippets/`](snippets/) — runnable, GPU-free demonstrations referenced from the chapters.

---

*Back to the [README / table of contents](README.md).*
