# RLFT reward-shape DOE — the *ranking* variant

**Question:** on a verifiable-math task, which reward shape best teaches an
otherwise-absent **output behavior** — and can we say so with confidence?

This is the redesign of [`../rlft-reward-shapes/`](../rlft-reward-shapes/README.md),
which swept the same four reward shapes and returned a **flat null result** (every
shape *and* the untuned baseline scored 1.000). That was an honest finding about a
saturated task, not a leaderboard. Here we re-engineer the *experiment* — **reusing
the orchestration unchanged** — so the shapes measurably diverge and can be ranked.

- **Example:** [`examples/run_doe_rlft_reward_ranking.py`](../../../examples/run_doe_rlft_reward_ranking.py)
- **Notebook:** [`notebooks/16_doe_reward_ranking.ipynb`](../../../notebooks/16_doe_reward_ranking.ipynb)
- **Experiment:** `geap-doe-rlft-reward-ranking`
- **Framework mechanics:** [`../../notes/doe-and-visualization.md` → "Making a reward-shape sweep actually *rank*"](../../notes/doe-and-visualization.md#making-a-reward-shape-sweep-actually-rank-the-ranking-variant)

## The load-bearing constraint: the "weaker base" lever doesn't help

The original plan for making this sweep rank had two levers — a **weaker base** (to
open *correctness* headroom) and a **neutral instruction** (to open *format*
headroom). The first lever collapsed, on empirical grounds first and a docs signal
second:

- **Empirically, every base we ran saturates correctness.** Two pilots on the
  *weakest* base (`gemini-2.5-flash-lite`) scored correctness ≈ 0.93–0.97 on a
  competition-flavored bank (see the pilot journey below), and `gemini-3.5-flash`
  scored 0.900. Stepping the base *down* did not open correctness headroom — the task
  is verifiable grade-school-to-moderate math, which small models already do well.
- **Docs signal (unverified here):** GEAP tuning docs list `gemini-3.5-flash` as the
  reinforcement-tuning base; a 2.5 base is not documented for RLFT (though this repo's
  launcher defaults to `gemini-2.5-flash`). We did **not** attempt a 2.5 RLFT *launch*
  to confirm whether it 400s — it is moot, because the base that is *definitely*
  RLFT-supported (`gemini-3.5-flash`, proven by the null-result sweep) already
  saturates correctness.

**Conclusion:** `correctness` cannot be made a rankable axis for RLFT on this task —
no reachable base has headroom on it. What *can* rank is the **`format_rate`** axis,
which the neutral instruction opens on any base. This DOE is therefore run on
`gemini-3.5-flash` (RLFT-supported and confirmed to have format headroom) and ranks
primarily on `format_rate` (with `explanation_quality` secondary); `correctness` is
reported as the saturated control.

## What changed, and why it can rank

The null result had five root causes; each maps to a fix:

| Root cause (old sweep) | Fix (this sweep) |
|---|---|
| `Answer: <n>` marker handed to the model for free | **Neutral system instruction** (`NEUTRAL_SYSTEM_INSTRUCTION`) drops the marker → **format headroom** (the load-bearing fix — this is the axis that ranks) |
| Base aces the task (no headroom) | Confronted, not fixed: every reachable base saturates correctness (0.90–0.97 across three pilots, weakest base included), so **correctness stays a control axis**, not a rankable one |
| Grade-school, single/two-step problems | **Harder, tiered bank** — `rlft/bench.py`, 150 problems (50/tier), answers computed in Python: `easy` control, `medium` multi-step arithmetic (two-unknown systems, chained % changes, work rate), `hard` number theory / combinatorics / series with **large** answers (kept for a larger balanced split; it does not create correctness headroom) |
| Tiny test set (n=6) | **Stratified split** → test n≈30, balanced across tiers |
| One binary headline + arbitrary `max` tie-break | **Multi-axis scoring** + **bootstrap 95% CIs** |

### The three measured axes

Each reward optimizes a different objective, so a single headline can't rank them.
`run_rlft_multimetric_eval` scores all three from one generation pass, plus a
per-difficulty correctness breakdown:

| Axis | What it measures | The reward that should top it |
|---|---|---|
| **format_rate** (primary — the one with headroom) | reply carries a parseable `Answer: <n>` marker, regardless of correctness | `string-match` (and `code-exec`, whose reward must parse the marker) |
| **explanation_quality** | 0–1 from an **offline LLM judge** (distinct model from the training autorater) | `autorater` |
| **correctness** (control — saturated) | right number anywhere in the reply (marker-agnostic) | near-ceiling for all shapes on this base |

The neutral system instruction is load-bearing: with the marker no longer free,
`string-match` has something real to teach on the `format_rate` axis, separating it
from `autorater` (which never rewards the marker).

## The pilot gate (before spending on four jobs)

The example scores the **untuned** base on the held-out test *first* and refuses to
launch unless the **format axis** has headroom — baseline `format_rate < 0.5` (the
marker is no longer free). `correctness` is printed but **not** gated: on
`gemini-3.5-flash` it is saturated by design. Run the gate without any tuning spend:

```bash
uv run python examples/run_doe_rlft_reward_ranking.py --pilot-only
```

`--force` launches even if the gate fails. This makes it a real experiment, not a
spend-first.

## Ranking methodology

- Rank shapes on **each** axis (`format_rate` primary); print the per-axis winner.
- Report correctness with a **bootstrap 95% CI** (`evaluate.bootstrap_ci`, seeded,
  stdlib-only) so any claimed gap comes with whether it is significant. Even n≈30
  gives wide CIs on a binary metric; overlapping CIs mean "not distinguishable yet",
  which is itself a finding.

## How to run

```bash
# pilot gate only — no tuning spend
uv run python examples/run_doe_rlft_reward_ranking.py --pilot-only

# pilot gate + (if format headroom) four RLFT jobs + leaderboard (incurs tuning cost)
uv run python examples/run_doe_rlft_reward_ranking.py

# same, plus write metrics.png + metrics_by_tier.png here (needs the viz group)
uv run --group viz python examples/run_doe_rlft_reward_ranking.py --plot

# launch even if the pilot gate fails
uv run python examples/run_doe_rlft_reward_ranking.py --force
```

Reruns are **idempotent** — jobs are reused by display name
(`geap-doe-rew-rank-<shape>-default`), so re-running only re-scores offline (a judge
429 mid-eval therefore costs a re-score, never re-tuning).

### Operational gotchas (inherited)

- **Eval must replay the training framing.** `run_rlft_multimetric_eval` threads each
  record's `systemInstruction` through to inference; dropping it measures a prompt
  mismatch, not the model (the bug that produced the earlier sweep's false 0.000). See
  [`../rlft-reward-shapes/README.md`](../rlft-reward-shapes/README.md).
- **Tuned-endpoint location / baseline routing.** A tuned Gemini 3.x model lands on a
  `us`/`eu` multi-region endpoint — route with `genai_client_for_endpoint`. The
  untuned base and the judge run on `global` (a separate client from the regional
  tuning one).
- **Autorater needs a fully-qualified judge path**
  (`projects/<p>/locations/<l>/publishers/google/models/gemini-2.5-flash`).
- **Judge circularity.** The explanation-quality judge (`gemini-2.5-pro`) must differ
  from the training autorater reward (`gemini-2.5-flash`), or it grades the model with
  its own trainer.
- **The judge can't disable thinking.** `gemini-2.5-pro` rejects `thinking_budget=0`
  (the `inference.generate` default) with a 400; the judge closure passes
  `thinking_budget=-1` (dynamic thinking). A flash/flash-lite judge would accept 0.
  Found at the pilot gate.

## Results

> **Status: re-targeted to `gemini-3.5-flash`; pilot cleared the format gate and the
> four jobs are launching.** The leaderboard below is pending the live run. The
> journey to this design (two prior pilots on a 2.5 base) is recorded first because it
> *is* the finding.

### The pilot journey — why correctness was abandoned as a rank axis

| Pilot | Base | Bank | correctness | format_rate | Gate |
|---|---|---|---|---|---|
| 1 | `gemini-2.5-flash-lite` | v1 (grade-school medium/hard) | 0.967 (29/30) | 0.000 | FAIL (correctness) |
| 2 | `gemini-2.5-flash-lite` | hardened (systems / number theory) | 0.933 (28/30) | 0.000 | FAIL (correctness) |
| 3 | **`gemini-3.5-flash`** | hardened | 0.900 (27/30) | **0.000** | **PASS (format)** |

Pilots 1–2 used the *weaker* base the plan called for and still aced correctness —
even on a competition-flavored bank, and even acing the number-theory "hard" tier.
Stepping the base down did not open correctness headroom, and GEAP docs point to
`gemini-3.5-flash` as the RLFT base anyway. So the re-target is the sound move: the
base that is definitely RLFT-supported also saturates correctness, so there is no
weaker-base escape. **Correctness has no headroom to rank on, regardless of base tier
or bank difficulty.**

### Pilot 3 — untuned `gemini-3.5-flash`, held-out n=30 (the launch gate)

| Axis | Score | Gate | Headroom? |
|---|---|---|---|
| `format_rate` | **0.000** (0/30) | < 0.50 | ✅ full |
| `correctness` | **0.900** (27/30) | not gated | saturated by design |

Per-tier correctness: easy **1.000**, medium **0.800**, hard **0.900**. The format
axis has *full* headroom — dropping the marker from the system instruction leaves the
base emitting it 0% of the time — so the sweep can rank shapes on `format_rate`.
**Gate PASS → four jobs launched.**

### Expected leaderboard (to be confirmed by the live run)

- `string-match` and `code-exec` should top **`format_rate`** (both rewards require
  the marker — string-match to score, code-exec to parse the answer it verifies).
- `autorater` should top **`explanation_quality`** and leave `format_rate` near
  baseline (it never rewards the marker).
- `correctness` should stay near ceiling for every shape (the documented control).
- Whether the `format_rate` gaps clear their bootstrap CIs at n≈30 is itself part of
  the finding — overlapping CIs would mean "not yet distinguishable."
