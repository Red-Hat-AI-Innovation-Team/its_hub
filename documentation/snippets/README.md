# Runnable snippets

Small, **GPU-free** programs that demonstrate the mechanics described in the
[documentation chapters](../README.md). Each uses the *real* `its_hub` code (real algorithms and real
helper functions) with tiny mock LMs / reward models, so what you see is exactly what the library does —
no model server, no API key, no GPU.

Run any of them with the project's conda env (see [Chapter 12](../12-running-it.md)):

```bash
PY=/home/exx/miniconda3/envs/epf/bin/python      # or:  conda run -n epf python
$PY documentation/snippets/epf_logweights_demo.py
$PY documentation/snippets/entropic_temperature_demo.py
$PY documentation/snippets/self_consistency_demo.py
```

| Snippet | Demonstrates | Chapter |
|---------|--------------|---------|
| [`epf_logweights_demo.py`](epf_logweights_demo.py) | PRM score → `logit` → softmax over particles == normalized **odds**; then runs the real `ParticleFiltering` and prints particle log-weights | [07](../07-particle-filtering.md) |
| [`entropic_temperature_demo.py`](entropic_temperature_demo.py) | the three temperature schedules (ESS / entropy / base), reproduces the test assertions, and shows how `T>1` flattens a collapsed weight distribution | [08](../08-entropic-particle-filtering.md) |
| [`self_consistency_demo.py`](self_consistency_demo.py) | the real `SelfConsistency` voting on `\boxed{}` answers extracted by a regex projection | [05](../05-self-consistency-and-best-of-n.md) |
| [`self_certainty_demo.py`](self_certainty_demo.py) | particle weights from the **generator's own logprobs/entropy** (no PRM): step logprobs → scalar → log-weight (styles `logit`/`raw`), then real `ParticleFiltering(weight_source='self_certainty')` | Part 2 experiment |

All four are verified to run in the `epf` env.
