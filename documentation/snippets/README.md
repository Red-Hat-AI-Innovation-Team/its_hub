# Runnable snippets

Small, **GPU-free** programs that demonstrate the mechanics described in the
[documentation chapters](../README.md). Each uses the *real* `its_hub` code (real algorithms and real
helper functions) with tiny mock LMs, so what you see is exactly what the library does —
no model server, no API key, no GPU.

Run any of them with the project's conda env (see [Chapter 12](../12-running-it.md)):

```bash
PY=/home/exx/miniconda3/envs/epf/bin/python      # or:  conda run -n epf python
$PY documentation/snippets/epf_logweights_demo.py
$PY documentation/snippets/entropic_temperature_demo.py
$PY documentation/snippets/self_certainty_demo.py
```

| Snippet | Demonstrates | Chapter |
|---------|--------------|---------|
| [`epf_logweights_demo.py`](epf_logweights_demo.py) | self-certainty confidence `s = exp(mean step logprob)` → `logit` → softmax over particles == normalized **odds**; then runs the real `ParticleFiltering` and prints particle log-weights | [07](../07-particle-filtering.md) |
| [`entropic_temperature_demo.py`](entropic_temperature_demo.py) | the three temperature schedules (ESS / entropy / base), reproduces the test assertions, and shows how `T>1` flattens a collapsed weight distribution | [08](../08-entropic-particle-filtering.md) |
| [`self_certainty_demo.py`](self_certainty_demo.py) | particle weights from the **generator's own logprobs/entropy** (the library's only weight source): step logprobs → scalar → log-weight (styles `logit`/`raw`), then the real `ParticleFiltering` end-to-end | Part 2 experiment |

All three are verified to run in the `epf` env.
