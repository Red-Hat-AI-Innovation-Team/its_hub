# Chapter 8 — Entropic Particle Filtering: The Namesake

> *Previous: [Particle Filtering](07-particle-filtering.md) · Next: [Putting It Together](11-putting-it-together.md)*

This repository is called **entropic-particle-filter** for the algorithm in this chapter. Everything so
far has been building toward it. **Entropic Particle Filtering (ePF)** is plain particle filtering plus
one carefully-placed idea: a **temperature** that *flattens the resampling distribution early on* so the
particle swarm doesn't collapse onto a single lucky trajectory before it has explored. The "entropic"
adjective is literal — one of the temperature schedules is driven by the **entropy** of the weight
distribution.

([`its_hub/core/algorithms/particle_filtering.py`](../its_hub/core/algorithms/particle_filtering.py),
class `EntropicParticleFiltering`, a subclass of `ParticleFiltering`.)

## The problem it solves: degeneracy and impoverishment

Recall from [Chapter 7](07-particle-filtering.md) that particle filters suffer **weight degeneracy** —
mass piling onto one particle — measured by a low **Effective Sample Size**, $\mathrm{ESS}=1/\sum_i p_i^2$.
Resampling a degenerate population produces many identical copies of one trajectory, which then diverge
only slightly: this second pathology is **sample impoverishment**. For long, multi-step math problems
this is fatal — the filter commits to one chain of reasoning at step 2 and can never recover if it was
wrong.

The classical SMC cure is **tempering**: soften the weights so resampling is less aggressive while
uncertainty is high, then sharpen them as the trajectories mature. ePF implements exactly this, but
*adaptively* — it only intervenes when it detects the swarm is collapsing, and only early in generation.

## The mechanism: temperature on the resampling softmax

The entire intervention lives in one override:
`EntropicParticleFiltering._weights_to_probabilities`, the hook the base class calls each outer step to
turn log-weights into the resampling distribution
([`particle_filtering.py`](../its_hub/core/algorithms/particle_filtering.py)):

```python
# its_hub/core/algorithms/particle_filtering.py — EntropicParticleFiltering
def _weights_to_probabilities(self, log_weights, current_step, num_particles):
    probabilities = _softmax(log_weights)
    temperature = self._temperature_annealing(probabilities, current_step, num_particles)
    # apply temperature annealing to the log weights
    return _softmax(np.asarray(log_weights) * (1 / temperature))   # tempered resampling distribution
```

(Plain `ParticleFiltering._weights_to_probabilities` is just `_softmax(log_weights)`; there is no
`does_entropic_annealing` flag anymore — subclassing *is* the switch.)

Mathematically, the resampling probabilities become a **tempered softmax** with temperature $T$:

$$
p_i(T) \;=\; \frac{\exp(w_i / T)}{\sum_j \exp(w_j / T)}.
$$

- $T = 1$: the ordinary distribution from [Chapter 7](07-particle-filtering.md).
- $T > 1$: **flatter** — differences between weights shrink, the distribution moves toward uniform,
  weak particles keep a fighting chance → **diversity preserved**.
- $T \to \infty$: uniform — every particle equally likely to survive.

Because $T \ge 1$ always (it's clamped, below), ePF only ever *flattens*, never sharpens beyond baseline.

## When does it intervene? The gate

Crucially, $T>1$ is applied **only when the swarm is actually collapsing, and only early**. The gate is
in `EntropicParticleFiltering._temperature_annealing`:

```python
# its_hub/core/algorithms/particle_filtering.py — _temperature_annealing
progress = current_step / self.max_steps
entropy_n = self._entropy_n(probabilities)
ess = self._effective_sample_size(probabilities)
ess_ratio = ess / num_particles

temperature = 1.0
if ess_ratio < self.ess_threshold and progress < self.early_phase:
    match self.temperature_method:
        case TemperatureMethod.ESS:     temperature = self._temperature_ess(ess_ratio, progress)
        case TemperatureMethod.ENTROPY: temperature = self._temperature_entropy(entropy_n, progress)
        case TemperatureMethod.BASE:    temperature = self._temperature_base(value_max, progress)
return temperature
```

Read the condition as English: *"if the swarm has collapsed (`ess_ratio` below `ess_threshold`,
default 0.5) **and** we're still in the early phase (`progress` below `early_phase`, default 0.5), then
crank up the temperature; otherwise leave it at 1."* Two intuitive knobs:

- **`ess_threshold`** — how collapsed must the swarm be before we intervene. Higher = intervene more
  readily.
- **`early_phase`** — how far into the trajectory (as a fraction of `max_steps`) we're still willing to
  intervene. After this point, let the filter sharpen normally and converge.

`progress` is just $t / T_{\max}$ — the fraction of `max_steps` consumed.

## The two ingredients: ESS and normalized entropy

Both are computed on the *current* softmax probabilities.

**Effective Sample Size** (`_effective_sample_size`):

$$ \mathrm{ESS} = \frac{1}{\sum_i p_i^2}, \qquad \mathrm{ess\_ratio} = \frac{\mathrm{ESS}}{N}\in(0,1]. $$

**Normalized entropy** (`_entropy_n`):

$$ H_n = \frac{-\sum_i p_i \ln p_i}{\ln N} \in [0,1]. $$

Both measure "how spread out is the weight?" — $1$ = perfectly uniform/healthy, near $0$ = collapsed.
The normalization by $\ln N$ (the max possible entropy for $N$ particles) makes $H_n$ comparable across
different particle counts.

## The three temperature schedules

All three return $T = \max(1.0, \text{value})$ so they can only flatten. They differ in what drives
`value`.

### 1. ESS-based (the ePF default) — `_temperature_ess`

$$ T_{\mathrm{ESS}} = \max\!\Big(1,\; \tfrac{1}{\mathrm{ess\_ratio}}\,(1-\mathrm{progress})\Big). $$

The more collapsed the swarm (smaller `ess_ratio`), the *larger* the temperature — directly proportional
to the severity of the collapse. The $(1-\mathrm{progress})$ factor anneals it toward $1$ as generation
proceeds. *Worked example from the test suite*
(`test_temperature_functions` in [`tests/test_entropic_annealing.py`](../tests/test_entropic_annealing.py)):
`ess_ratio=0.2, progress=0.2` ⟹ $\frac{1}{0.2}(1-0.2) = 5 \times 0.8 = 4.0$;
`ess_ratio=0.5, progress=0.8` ⟹ $2 \times 0.2 = 0.4 \to \max(1, 0.4) = 1.0$.

### 2. Entropy-based — `_temperature_entropy`

$$ \beta = H_n + (1 - H_n)\,\mathrm{progress}, \qquad T_{H} = \max\!\Big(1, \tfrac{1}{\beta}\Big). $$

When entropy is high ($H_n\to 1$), $\beta\to 1$ and $T\to 1$ — no need to intervene, the swarm is already
diverse. When entropy is low ($H_n\to 0$) *and* it's early ($\mathrm{progress}\to 0$), $\beta\to 0$ and
$T$ grows large — strong flattening. As `progress` $\to 1$, $\beta\to 1$ regardless, annealing $T$ back
to baseline. This is the schedule that most literally earns the name *entropic*. *Test*
(`test_temperature_functions`): `H_n=0.5,
progress=0.3` ⟹ $1/(0.5 + 0.5\cdot0.3)=1/0.65\approx1.538$; `H_n=1.0` ⟹ $T=1.0$.

### 3. Base (linear) — `_temperature_base`

$$ T_{\mathrm{base}} = \max\!\big(1,\; v_{\max} - \mathrm{progress}\big), \quad v_{\max}=2.0 \text{ by default}. $$

The simplest: ignore the swarm's state, just decay linearly from $v_{\max}$ toward $1$ as generation
proceeds. *Test* (`test_temperature_functions`):
`v_max=2.0, progress=0.5` ⟹ $1.5$; `v_max=0.8, progress=0.5` ⟹ $\max(1, 0.3)=1.0$.

### The shape, sketched

```text
  T │
4.0 ┤ ●  ESS (ess_ratio=0.2)          ← severe collapse early ⇒ strong flattening
    │  \
    │   \                              All schedules:
2.0 ┤ ●··●·· BASE (v_max=2)            • start high while early & collapsed
    │  \   \···                        • clamp at T = 1 (never sharpen)
1.5 ┤ ●  ●    ENTROPY                  • anneal toward 1 as progress → early_phase
    │   ` ·,___
1.0 ┤────────────●───●───●────────────► progress (t / max_steps)
    0          early_phase=0.5        1.0
   (beyond early_phase OR ess_ratio ≥ threshold ⇒ T = 1, i.e. ordinary PF)
```

## How `EntropicParticleFiltering` is wired

The subclass fixes the right defaults (constructor in
[`particle_filtering.py`](../its_hub/core/algorithms/particle_filtering.py)):

```python
EntropicParticleFiltering(
    sg,
    final_response_selection=SelectionMethod.ARGMAX,
    resampling_method=ResamplingMethod.SYSTEMATIC,   # lower-variance, diversity-friendly
    temperature_method=TemperatureMethod.ESS,        # the adaptive default ("ess"|"entropy"|"base")
    ess_threshold=0.5,
    early_phase=0.5,
    self_certainty_signal="mean_logprob",            # or "entropy"  (passed through to the base class)
    self_certainty_style="logit",                    # or "raw"
    top_logprobs=None,
)
```

So ePF = `ParticleFiltering` with **systematic** resampling, the **ESS** schedule, and the tempered
`_weights_to_probabilities` override. (No `prm=` — the weights come from self-certainty, exactly as in
Chapter 7.) Everything else — the logit weights, the softmax, the
`ParticleFilteringResult`, `the_one` — is inherited unchanged from [Chapter 7](07-particle-filtering.md).
Note ePF flattens **only the resampling distribution**; the underlying log-weights (and thus the final
`ARGMAX` selection) are not tempered — final selection re-softmaxes the raw log-weights.

## Why this matters

On easy problems, ePF behaves like plain PF — `ess_ratio` stays healthy, the gate never fires, $T=1$. On
**hard, long** problems where naive PF would prematurely commit, ePF detects the early collapse and holds
the swarm open long enough to find the good reasoning path. This is the algorithmic contribution the
repository is named for; this chapter is the mechanism.

A runnable demo that evaluates all three schedules across `progress` values (mirroring the test
assertions, no GPU needed) is in
[`snippets/entropic_temperature_demo.py`](snippets/entropic_temperature_demo.py).

---

*Next: [Chapter 11 — Putting It Together](11-putting-it-together.md): one end-to-end trace and the
mental model.*
