# Lattice check schema and materialization contract

This documents how a `kind: check` rule becomes **deterministic, programmable
enforcement** — the core goal of lattice (SPEC §5.3 Check installation).
Instruction rules (`kind: instruction`) are unaffected; they render into the
CLAUDE.md managed block as before.

Audience: contributors editing `.lattice/packs/`. This file documents the
current schema and the prose→enforcement pipeline; it does not carry project
history.

## Check rule fields

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | Stable id, e.g. `PYTHON-CHK-001`. |
| `kind` | yes | `check`. |
| `statement` | yes | Human-readable intent of the check. |
| `scope` | yes | Path glob the check applies to. Used for freshness (§5.4) and scope-aware triggering (§5.3). |
| `state` | yes | `warn-only` (record, never block) or `enforced` (block per tier). |
| `tier` | yes | Budget class — see below. |
| `check_command` | for installable checks | The **deterministic shell command** the runner executes. Its exit status is the check result. Absent = not directly installable (protocol/action check); the installer skips it with a logged reason. |
| `expected_exit` | no (default `0`) | Exit status that counts as pass. |
| `provenance` | yes | Evidence, strongest-first (SPEC §3.3). |

## The materialization contract

This is the whole pipeline, and the part a meta-harness must reproduce:

```
.lattice/packs/*.yaml          prose rules, human-authored/mined
        │
        │  install_checks.py          ← build time; run when packs change
        │    1. iter_checks()           collect every kind: check
        │    2. freshness filter        drop what this repo cannot run (§5.4)
        │    3. group by tier           inner / gate / deep
        │    4. dedupe by command       identical commands render one step
        │    5. render deterministically
        ▼
generated artifacts (committed, except the git hook)
   inner → .claude/settings.json  PostToolUse hook   (agent edits)
   inner → .git/hooks/pre-commit                     (human commits, per-clone)
   gate  → .github/workflows/lattice-checks.yaml     (pre-merge CI)
   deep  → .github/workflows/lattice-deep.yaml       (scheduled)
        │
        │  every artifact invokes the same runner
        ▼
run_check.py  ← run time
    executes check_command once per unique command
    compares exit status to expected_exit
    emits one T2 event per rule (§6.1)
    warn-only → always exit 0 │ enforced → exit 1 on mismatch
        ▼
.lattice/traces/T2/<date>.jsonl   evidence for the improvement loop (§7)
```

Two invariants make this work:

1. **Deterministic rendering.** Same lattice version → byte-identical artifacts.
   This is what makes drift detectable at all (see `CORE-CHK-001`).
2. **One runner, every path.** CI, the agent hook, and the pre-commit hook all
   call `run_check.py`, so a check behaves identically wherever it fires and
   every path produces comparable T2 records. Only the `--event` label differs.

## Tiers (SPEC §5.3) — budget class, not a location

A tier answers *how much time may this check take*, which then determines where
it can be installed. One tier may have **several** install targets: `inner`
installs twice, so enforcement does not depend on whether a human or an agent
made the edit.

| Tier | Installs as | Budget | On failure when `enforced` |
|---|---|---|---|
| `inner` | PostToolUse hook (`.claude/settings.json`) **and** `.git/hooks/pre-commit` | ~2 min | agent/committer must fix before proceeding |
| `gate` | pre-merge CI (`.github/workflows/lattice-checks.yaml`) | uncapped | merge blocked |
| `deep` | scheduled job (`.github/workflows/lattice-deep.yaml`) | uncapped | surfaced for the next improvement-loop pass |

Tier follows the budget: sub-second file-local checks (ruff) belong at `inner`;
whole-repo checks (pytest, build) at `gate`; anything whose purpose is to
accumulate a trend rather than guard a merge (coverage, dependency advisories)
at `deep`.

`.git/hooks/pre-commit` is per-clone and untracked, so the installer writes it
but drift verification never checks it. It refuses to overwrite a pre-existing
non-lattice hook.

## State semantics

- `warn-only` — the runner records a T2 event (SPEC §6.1) and **always exits 0**;
  nothing blocks. This is the default for every rule at v0 (Axiom A5, evidence
  before enforcement).
- `enforced` — the runner exits non-zero when `check_command`'s exit status
  differs from `expected_exit`, blocking per the tier. Promotion `warn-only →
  enforced` happens through the improvement loop (SPEC §7) on accumulated
  T2/T3 evidence — never by hand-editing this field without a manifest entry.

## Freshness verification (SPEC §5.4)

At install time every installable check must reference paths that exist in
*this* repository. A check whose `scope` directory is absent (e.g. the Rust and
Praxis packs' `rust/**`, or a `notebooks/**` check with no `notebooks/`), or
which carries no `check_command`, **fails the install and is skipped** — it is
never silently materialized. This is the recipient-side gate that keeps grafted
packs (SPEC §4.4) from installing checks the recipient cannot run.

**Known limit:** freshness verifies *paths*, not *environmental preconditions*.
A check can pass freshness and still be unrunnable because it needs a service, a
credential, or a specific runner. `tests/e2e` is the worked example in this repo
— the directory exists, so freshness passes, but the suite needs a live vLLM
endpoint and a self-hosted runner. It was therefore never given a
`check_command`.

## Command self-containment

A `check_command` must carry its own environment precondition. Commands mined
verbatim from CI inherit CI's implicit setup steps and silently misbehave
elsewhere.

Worked example: `uv run pytest tests/` is exactly what `.github/workflows/tests.yaml`
runs, and it is correct *there* because a preceding step runs `uv sync --extra dev`.
Run locally against an unsynced `.venv`, `uv run` fell through to a PATH-resolved
pyenv-global pytest under a different interpreter and reported
`ModuleNotFoundError: No module named 'its_hub'` — a false failure blamed on the
repository. The stored command is therefore
`uv run --extra dev python -m pytest tests/ --ignore=tests/e2e`: `--extra dev`
guarantees the tool is in the project environment, and `python -m` fails loudly
instead of resolving a foreign binary.

## Drift (`CORE-CHK-001`)

Editing a pack without re-running the installer leaves CI executing a stale step
list, so every other check quietly stops meaning what it claims. `--verify`
re-renders the artifacts in memory and diffs them against what is committed,
exiting 1 on any mismatch. It runs as a GATE check, so a PR that changes rules
without materializing them fails.

Because the version hash covers **all** pack bytes and is stamped into each
generated file, *any* pack edit — including a comment-only change to a
provenance note — requires re-running the installer. That is deliberate: it
guarantees a committed artifact always corresponds to exactly the packs beside
it, at the cost of some churn.

## Tooling

- `.lattice/bin/_common.py` — pack loading, version hashing, freshness. Shared by
  both scripts so the build-time and run-time views of "what is a check" agree.
- `.lattice/bin/run_check.py` — the runtime runner.
  `run_check.py ID [ID ...]` or `run_check.py --tier {inner,gate,deep}`, with
  `--event LABEL` for the trace context. Emits T2; honors warn-only vs enforced.
- `.lattice/bin/install_checks.py` — the materializer.
  Bare invocation writes the artifacts; `--verify` writes nothing and exits 1 on
  drift. Prints an install/skip report and never silently drops a rule.

Both are invoked as `uv run --with pyyaml python <script>` so PyYAML never
becomes a project dependency (CORE-INS-007).
