# Chapter 6 — Beam Search: Deterministic Step-by-Step Search

> *Previous: [The Simple Scalers](05-self-consistency-and-best-of-n.md) · Next: [Particle Filtering](07-particle-filtering.md)*

Beam Search is the bridge between the simple scalers and the probabilistic core. Like Particle
Filtering, it grows solutions **one step at a time** and uses a **process reward model** to judge each
step. Unlike Particle Filtering, it makes its decisions **deterministically** — sort by score, keep the
top few. Reading Beam Search first makes the particle filter's one twist (resampling) stand out in sharp
relief.

([`its_hub/core/algorithms/beam_search.py`](../its_hub/core/algorithms/beam_search.py))

## The mental model

A *beam* is a partial solution path. At each level we extend every live beam by one step, score the
extended beams with the PRM, **keep the top `beam_width`**, and then **clone** those survivors back up to
the full population. Repeat until everything stops.

```text
                       budget = 6, beam_width = 2  →  num_beams = 3
 level 0:  [ b1 ][ b2 ][ b3 ]        (3 empty paths)
              │     │     │
   extend &   ▼     ▼     ▼
   PRM-score 0.7   0.4   0.9   ...   (each grown one step, scored on the WHOLE prefix)
                 sort ↓ keep top beam_width=2
   survivors:  [0.9][0.7]
                 clone back to num_beams=3:
 level 1:  [0.9'][0.7'][0.9''] ...   (deepcopies; each grows independently next level)
              ...
 stop when all paths hit max_steps or the stop_token  →  argmax(score) wins
```

## The data structures

A beam is a `Path` — steps, a stopped flag, and a single scalar `score`
([`beam_search.py:29-41`](../its_hub/core/algorithms/beam_search.py#L29-L41)):

```python
@dataclass
class Path:
    steps: list[str]
    is_stopped: bool
    score: float
    def deepcopy(self): ...   # independent copy so clones diverge
```

Compare this to the particle filter's `Particle` (next chapter): a `Particle` carries a *list* of
`partial_log_weights`; a `Path` carries a single `score`. That difference is the whole story — Beam
Search keeps one number and sorts on it; the particle filter keeps a weight *history* and resamples on
it.

## Budget → search shape

```python
# its_hub/core/algorithms/beam_search.py:141-150
assert budget % self.beam_width == 0, "budget must be divisible by beam_width"
assert budget >= self.beam_width, "budget must be greater than or equal to beam_width"
num_beams = budget // self.beam_width
candidates = [Path(steps=[], is_stopped=False, score=0) for _ in range(num_beams)]
```

So **`budget = num_beams × beam_width`**. `beam_width` is how many survivors you keep each level (search
*breadth* at the bottleneck); `num_beams` is how many candidates you carry overall. With `budget=12,
beam_width=4` you keep the top 4 each level and maintain 3× that population.

## One level: extend → score → (later) prune

`_asearch_one_level` does the per-level work
([`beam_search.py:55-113`](../its_hub/core/algorithms/beam_search.py#L55-L113)). It:

1. Collects the non-stopped beams and **batch-generates** their next step via `StepGeneration.aforward`
   ([`beam_search.py:74-87`](../its_hub/core/algorithms/beam_search.py#L74-L87)).
2. **Batch-scores** the extended beams with the PRM — crucially, scoring the **whole prefix so far**,
   reassembled by `_post_process(..., stopped=True)`:

```python
# its_hub/core/algorithms/beam_search.py:97-111
scores = await self.prm.ascore(
    prompt,
    [self.sg._post_process(steps_so_far_per_prompt, stopped=True)
     for steps_so_far_per_prompt in steps_so_far],
)
# ...
c.score = scores[i]     # the beam's score is REPLACED with the latest whole-prefix score
```

Note `c.score = scores[i]` is an **assignment, not an accumulation**. Each level the beam's score is the
PRM's judgment of the entire partial solution to date — exactly like the particle filter (Chapter 7),
which also re-scores the full prefix each step.

## The main loop: sort, keep top-k, clone

```python
# its_hub/core/algorithms/beam_search.py:152-171
while not all(c.is_stopped for c in candidates):
    candidates = await self._asearch_one_level(lm, candidates, chat_messages.to_prompt(), ...)
    candidates.sort(key=lambda x: x.score, reverse=True)   # best first
    candidates = candidates[: self.beam_width]             # PRUNE to top beam_width
    new_candidates = []                                    # CLONE back to num_beams
    for _ in range(num_beams):
        for c in candidates:
            new_candidates.append(c.deepcopy())
    candidates = new_candidates
```

This is pure, greedy selection: **no randomness, no weights**. The survivors are duplicated so the
search keeps exploring multiple continuations of the best prefixes. Because clones are `deepcopy`-d, they
diverge independently on the next level.

## Stopping and the final answer

The loop ends when **all** candidates are stopped — and "stopped" is decided inside `StepGeneration`
(`max_steps` reached or `stop_token` seen), not by the model. The winner is a plain `argmax` over the
final scores ([`beam_search.py:173-187`](../its_hub/core/algorithms/beam_search.py#L173-L187)):

```python
result = BeamSearchResult(
    responses=[{"role": "assistant", "content": self.sg._post_process(c.steps, stopped=True)} for c in candidates],
    scores=scores,
    selected_index=int(np.argmax(scores)),
    steps_used=steps_used,
)
```

## Beam Search vs Particle Filtering — the one-line contrast

They share *almost* everything — `StepGeneration`, a PRM scoring the whole prefix each step, a population
of growing trajectories. The difference is the selection rule:

| | Beam Search | Particle Filtering |
|---|---|---|
| Per-trajectory state | one scalar `score` | a list of `partial_log_weights` |
| Reward → decision | sort, keep **top-k** (greedy) | softmax → **resample** (stochastic) |
| Uses log-weights? | **no** — raw scores, compared directly | **yes** — logit of the score (Ch. 7) |
| Can a low-scoring path survive? | only if it's in the top-k | **yes**, with probability ∝ its weight |
| Randomness | none (argmax) | resampling + optional annealing |

That last row is why particle filtering can escape the "greedy gets stuck" failure mode: a path that
looks mediocre now but leads somewhere good still has a *chance* to survive. Beam search would have
pruned it. We make this precise next.

---

*Next: [Chapter 7 — Particle Filtering](07-particle-filtering.md), where reward becomes log-weight.*
