# Chapter 99 — Glossary & References

> *Previous: [Running It](12-running-it.md) · Back to [README](README.md)*

## Glossary

**Inference-time scaling (ITS) / test-time compute.** Spending extra compute *at inference* (multiple
samples, search) to improve answer quality from a *fixed* model, rather than retraining. The premise of
the whole library ([Chapter 1](01-the-problem.md)).

**`budget`.** The integer compute allowance passed to every algorithm; its meaning is algorithm-specific
(see the [cheat-sheet](README.md#the-budget-cheat-sheet)).

**`the_one`.** The single chosen response dict returned by every result object
([`api/algorithm.py:8-25`](../its_hub/api/algorithm.py#L8-L25)).

**`ainfer` / `infer`.** The async primary entry point and its synchronous `asyncio.run` wrapper
([Chapter 2](02-architecture.md)).

**Process Reward Model (PRM).** A reward model that scores a *partial* reasoning trajectory, step by
step. Returns probability-like scores in $[0,1]$ (higher = better). Used by Beam Search and Particle
Filtering. Interface: [`api/reward_models/prm.py`](../its_hub/api/reward_models/prm.py)
([Chapter 4](04-reward-models.md)).

**Outcome Reward Model (ORM).** A reward model that scores a *complete* conversation. Built-in example:
`LLMJudge` (an LLM grading on a 0–10 scale). Used by Best-of-N. Interface:
[`api/reward_models/orm.py`](../its_hub/api/reward_models/orm.py).

**Process vs outcome supervision.** Judging the reasoning *as it unfolds* vs. judging only the final
answer. Process supervision can prune bad paths early but needs a step-level model (Lightman et al.
2023).

**`aggregation_method`.** How `LocalVllmProcessRewardModel` collapses per-step PRM scores into one
number: `"prod"` (multiply), `"mean"` (average), `"last"` (final step). **The only place per-step
rewards are multiplied** ([Chapter 4](04-reward-models.md), [Chapter 7](07-particle-filtering.md)).

**Particle.** One candidate reasoning trajectory in the particle filter — its steps, a stopped flag, and
a history of log-weights ([`particle_gibbs.py:44-63`](../its_hub/core/algorithms/particle_gibbs.py#L44-L63)).

**Log-weight.** A particle's weight in log-space. Here it is the **logit** of the PRM score:
$w=\operatorname{logit}(s)=\ln\frac{s}{1-s}$, computed by `_inv_sigmoid`
([`particle_gibbs.py:66-70`](../its_hub/core/algorithms/particle_gibbs.py#L66-L70)). See
[Chapter 7](07-particle-filtering.md) for the full derivation.

**Logit / inverse sigmoid.** $\operatorname{logit}(s)=\ln\frac{s}{1-s}$, the inverse of the sigmoid.
Maps a probability $[0,1]$ to a log-odds on the whole real line.

**Sequential Monte Carlo (SMC) / particle filter.** A family of algorithms that approximate a target
distribution with a weighted set of samples, evolved by *sample → weight → resample*. The probabilistic
core of this library (Gordon et al. 1993; Doucet et al. 2001).

**Importance sampling.** Estimating properties of a target distribution using samples from an easier
proposal, corrected by weights. Resampling weights in the particle filter are importance weights.

**Resampling.** Drawing a new particle population with replacement in proportion to weights — clones
strong particles, drops weak ones. `multinomial` (i.i.d.) or `systematic` (low-variance comb)
([`particle_gibbs.py:308-353`](../its_hub/core/algorithms/particle_gibbs.py#L308-L353)).

**Effective Sample Size (ESS).** $\mathrm{ESS}=1/\sum_i p_i^2 \in [1,N]$. A diagnostic for weight
concentration; low ESS signals degeneracy ([`particle_gibbs.py:228-243`](../its_hub/core/algorithms/particle_gibbs.py#L228-L243)).

**Particle degeneracy.** The pathology where weight concentrates on one particle (ESS → 1). **Sample
impoverishment** is the follow-on loss of diversity after resampling clones it. The problem ePF fights
([Chapter 8](08-entropic-particle-filtering.md)).

**Normalized entropy.** $H_n = -\sum_i p_i\ln p_i / \ln N \in [0,1]$; another spread diagnostic, used by
ePF's entropy temperature schedule ([`particle_gibbs.py:203-226`](../its_hub/core/algorithms/particle_gibbs.py#L203-L226)).

**Entropic annealing / tempering.** Raising the resampling **temperature** $T>1$ to flatten the
distribution (preserve diversity) early, annealing back to $T=1$ later. Tempered softmax
$p_i(T)=\frac{e^{w_i/T}}{\sum_j e^{w_j/T}}$ ([Chapter 8](08-entropic-particle-filtering.md)).

**Reference particle.** In Particle Gibbs (`num_iterations>1`), a surviving trajectory carried into the
next sweep as a fixed anchor, making the sweeps an MCMC chain
([Chapter 9](09-particle-gibbs.md)).

**Orchestrator.** The component that fans out parallel LM calls with a concurrency cap
(`LMOrchestrator` uses `asyncio.TaskGroup` + a thread-safe semaphore). Used by Self-Consistency and
Best-of-N ([Chapter 3](03-generating-text.md)).

**StepGeneration.** The adapter that turns the LM into a one-step-at-a-time generator for Beam Search and
Particle Filtering, stopping on `max_steps` or a `stop_token`
([`lms/step_generation.py`](../its_hub/core/lms/step_generation.py)).

**Projection function.** In Self-Consistency, the map from a response to the value voted on (e.g. the
boxed answer). Built via `create_regex_projection_function`
([`self_consistency.py:316-375`](../its_hub/core/algorithms/self_consistency.py#L316-L375)).

**PF / ePF / PG / PGAS.** Particle Filtering / Entropic PF / Particle Gibbs / PG with Ancestor Sampling —
the four members of the `ParticleGibbs` family ([Chapter 9](09-particle-gibbs.md)). PGAS is scaffolded
but `NotImplementedError`.

## References

### The paper this repository implements

- **Puri, I., Sudalairaj, S., Xu, G., Xu, K., & Srivastava, A. (2025).** *A Probabilistic Inference
  Approach to Inference-Time Scaling of LLMs using Particle-Based Monte Carlo Methods* (a.k.a. *"Rollout
  Roulette"*). arXiv:2502.01618. NeurIPS 2025.
  <https://arxiv.org/abs/2502.01618> · project page: <https://probabilistic-inference-scaling.github.io/>
  > **Direct lineage:** co-author **Kai Xu** is the author of `its_hub` (see
  > [`pyproject.toml`](../pyproject.toml)). This library is the reference implementation of that paper;
  > the particle-filtering machinery in [Chapter 7](07-particle-filtering.md) is the paper's method, and
  > the example in [Chapter 12](12-running-it.md) reproduces its headline setup (Qwen2.5-Math-1.5B-Instruct
  > + Qwen2.5-Math-PRM-7B).

### Inference-time scaling & verification

- **Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., et al. (2022).** *Self-Consistency Improves Chain
  of Thought Reasoning in Language Models.* arXiv:2203.11171. — basis for
  [Chapter 5](05-self-consistency-and-best-of-n.md). <https://arxiv.org/abs/2203.11171>
- **Cobbe, K., Kosaraju, V., et al. (2021).** *Training Verifiers to Solve Math Word Problems* (GSM8K).
  arXiv:2110.14168. — the "generate many, rerank with a verifier" recipe behind Best-of-N.
  <https://arxiv.org/abs/2110.14168>
- **Lightman, H., Kosaraju, V., Burda, Y., et al. (2023).** *Let's Verify Step by Step.* arXiv:2305.20050.
  — process supervision / PRMs ([Chapter 4](04-reward-models.md)). <https://arxiv.org/abs/2305.20050>
- **Uesato, J., Kushman, N., et al. (2022).** *Solving Math Word Problems with Process- and Outcome-Based
  Feedback.* arXiv:2211.14275. — process vs. outcome reward.
  <https://arxiv.org/abs/2211.14275>
- **Snell, C., Lee, J., Xu, K., & Kumar, A. (2024).** *Scaling LLM Test-Time Compute Optimally can be More
  Effective than Scaling Model Parameters.* arXiv:2408.03314. — the test-time-compute thesis.
  <https://arxiv.org/abs/2408.03314>

### Sequential Monte Carlo foundations

- **Gordon, N. J., Salmond, D. J., & Smith, A. F. M. (1993).** *Novel approach to nonlinear/non-Gaussian
  Bayesian state estimation.* IEE Proceedings F, 140(2), 107–113. — the bootstrap particle filter
  (sample → weight → resample).
- **Doucet, A., de Freitas, N., & Gordon, N. (eds.) (2001).** *Sequential Monte Carlo Methods in
  Practice.* Springer. — the standard SMC reference.
- **Andrieu, C., Doucet, A., & Holenstein, R. (2010).** *Particle Markov Chain Monte Carlo Methods.*
  J. R. Statist. Soc. B, 72(3), 269–342. — Particle Gibbs ([Chapter 9](09-particle-gibbs.md)).
- **Lindsten, F., Jordan, M. I., & Schön, T. B. (2014).** *Particle Gibbs with Ancestor Sampling.* JMLR,
  15, 2145–2184. — the PGAS refinement scaffolded in the code.
- **Neal, R. M. (2001).** *Annealed Importance Sampling.* Statistics and Computing, 11(2), 125–139. —
  tempering, the idea behind entropic annealing ([Chapter 8](08-entropic-particle-filtering.md)).
- **Del Moral, P., Doucet, A., & Jasra, A. (2006).** *Sequential Monte Carlo Samplers.* J. R. Statist.
  Soc. B, 68(3), 411–436. — adaptive tempering and ESS-based resampling.

### In-repo documents

- [`docs/algorithms.md`](../docs/algorithms.md) — the user-facing algorithm overview (conceptual).
- [`docs/PLANNING_WRAPPER.md`](../docs/PLANNING_WRAPPER.md) — planning wrapper user docs.
- [`docs/orchestration.md`](../docs/orchestration.md) — orchestrator / gateway integration.
- [`CLAUDE.md`](../CLAUDE.md) — developer commands and conventions.

---

*Back to the [README / table of contents](README.md).*
