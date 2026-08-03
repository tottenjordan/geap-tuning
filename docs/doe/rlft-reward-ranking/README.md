# RLFT reward-shape DOE — the *ranking* variant

**Question:** for a verifiable-math task **with genuine headroom**, which reward
function produces the best tuned model — and can we say so with confidence?

This is the redesign of [`../rlft-reward-shapes/`](../rlft-reward-shapes/README.md),
which swept the same four reward shapes and returned a **flat null result** (every
shape *and* the untuned baseline scored 1.000). That was an honest finding about a
saturated task, not a leaderboard. Here we re-engineer the *experiment* — **reusing
the orchestration unchanged** — so the shapes measurably diverge and can be ranked.

- **Example:** [`examples/run_doe_rlft_reward_ranking.py`](../../../examples/run_doe_rlft_reward_ranking.py)
- **Notebook:** [`notebooks/16_doe_reward_ranking.ipynb`](../../../notebooks/16_doe_reward_ranking.ipynb)
- **Experiment:** `geap-doe-rlft-reward-ranking`
- **Framework mechanics:** [`../../notes/doe-and-visualization.md` → "Making a reward-shape sweep actually *rank*"](../../notes/doe-and-visualization.md#making-a-reward-shape-sweep-actually-rank-the-ranking-variant)

## What changed, and why it can rank

The null result had five root causes; each maps to a fix:

| Root cause (old sweep) | Fix (this sweep) |
|---|---|
| Base aces the task (no headroom) | **Weaker base** — `gemini-2.5-flash-lite`, not `gemini-3.5-flash` |
| Grade-school, single/two-step problems | **Harder, tiered bank** — `rlft/bench.py`, 150 problems (50/tier), answers computed in Python: `easy` is a near-ceiling control, `medium` is multi-step arithmetic (two-unknown systems, chained % changes, work rate), `hard` is **number theory / combinatorics / series** (modular exponentiation, constrained combinations, geometric sums) with deliberately **large** answers |
| `Answer: <n>` marker handed to the model for free | **Neutral system instruction** (`NEUTRAL_SYSTEM_INSTRUCTION`) drops the marker → format headroom |
| Tiny test set (n=6) | **Stratified split** → test n≈30, balanced across tiers |
| One binary headline + arbitrary `max` tie-break | **Multi-axis scoring** + **bootstrap 95% CIs** |

### The three measured axes

Each reward optimizes a different objective, so a single headline can't rank them.
`run_rlft_multimetric_eval` scores all three from one generation pass, plus a
per-difficulty correctness breakdown:

| Axis | What it measures | The reward that should top it |
|---|---|---|
| **correctness** (primary) | right number anywhere in the reply (marker-agnostic) | `code-exec` / `composite` |
| **format_rate** | reply carries a parseable `Answer: <n>` marker, regardless of correctness | `string-match` |
| **explanation_quality** | 0–1 from an **offline LLM judge** (distinct model from the training autorater) | `autorater` |

The neutral system instruction is load-bearing: with the marker no longer free,
`string-match` has something real to teach on the `format_rate` axis, separating it
from the correctness rewards.

## The pilot gate (before spending on four jobs)

The example scores the **untuned** base on the held-out test *first* and refuses to
launch unless there is real headroom — baseline `correctness` < 0.7 **and**
`format_rate` < 0.5 (the marker is no longer free). If the base still aces it, pick a
weaker base or a harder tier before launching; `--force` overrides. This makes it a
real experiment, not a spend-first.

## Ranking methodology

- Rank shapes on **each** axis (correctness primary); print the per-axis winner.
- Report correctness with a **bootstrap 95% CI** (`evaluate.bootstrap_ci`, seeded,
  stdlib-only) so "best shape" comes with whether the gap over the runner-up /
  baseline is significant — not a coin-flip. Even n≈30 gives wide CIs on a binary
  metric; overlapping CIs mean "not distinguishable yet", which is itself a finding.

## How to run

```bash
# pilot gate + (if headroom) four RLFT jobs + leaderboard (incurs tuning cost)
uv run python examples/run_doe_rlft_reward_ranking.py

# same, plus write metrics.png + metrics_by_tier.png here (needs the viz group)
uv run --group viz python examples/run_doe_rlft_reward_ranking.py --plot

# launch even if the pilot gate fails (e.g. to inspect a marginal base)
uv run python examples/run_doe_rlft_reward_ranking.py --force
```

Reruns are **idempotent** — jobs are reused by display name
(`geap-doe-rew-rank-<shape>-default`), so re-running only re-scores offline.

### Operational gotchas (inherited)

- **Eval must replay the training framing.** `run_rlft_multimetric_eval` threads each
  record's `systemInstruction` through to inference; dropping it measures a prompt
  mismatch, not the model (the bug that produced the earlier sweep's false 0.000). See
  [`../rlft-reward-shapes/README.md`](../rlft-reward-shapes/README.md).
- **Tuned-endpoint location / baseline routing.** A tuned model lands on the
  endpoint's own (multi-)region — route with `genai_client_for_endpoint`. The untuned
  base and the judge run on `global` (Gemini 2.x inference), a separate client from
  the regional tuning one.
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

> **Status: pilot run — the gate fired and the four jobs were *not* launched.** The
> bank, multi-axis eval, bootstrap CIs, pilot gate, example, notebook, and tests are
> implemented and unit-tested; the pilot then scored the untuned base on the held-out
> test split (n=30) *before* any tuning spend.

### Pilot result — untuned `gemini-2.5-flash-lite`, held-out n=30

| Axis | Score | Gate ceiling | Headroom? |
|---|---|---|---|
| `correctness` | **0.967** (29/30) | < 0.70 | ❌ none |
| `format_rate` | **0.000** (0/30) | < 0.50 | ✅ full |

Per-tier correctness: easy **1.000** (10/10), medium **1.000** (10/10), hard
**0.900** (9/10). (No `explanation_quality` — the gate needs only correctness and
format, so the pilot skips the judge to save calls.)

**Gate: FAIL → no launch.** The primary axis (`correctness`) is still saturated: even
the weaker base solves the harder tiered bank, missing only one hard problem. The
`format_rate` axis, by contrast, has *full* headroom — dropping the `Answer: <n>`
marker from the system instruction (the neutral-instruction change) leaves the base
emitting it 0% of the time, exactly as designed.

**This is the pilot gate working as intended** — it caught the ceiling for the price
of ~30 base-model generations instead of four RLFT jobs. It also sharpens the earlier
[cautionary tale](../rlft-reward-shapes/README.md): the null result was **not** only
about the strong `gemini-3.5-flash` base or the tiny n=6 split. A verifiable
grade-school-to-moderate math task is saturated even by a small 2.5 model, so
**correctness has no headroom to rank on regardless of base tier.** Making that axis
discriminate needs a genuinely harder task — competition-grade multi-step problems,
or a domain the base has not memorized — not merely a weaker base.

**What would still rank (single-axis).** Because `format_rate` has full headroom, a
launch *would* be expected to separate `string-match` (and `code-exec`, whose reward
must parse the marker to verify) from `autorater` (which never rewards the marker) on
that one axis. That is a real but one-dimensional result; the multi-axis ranking the
design targets needs the harder bank above.

### Update — bank hardened after this pilot

The pilot numbers above describe the **first-iteration** `rlft/bench.py`, whose
`medium`/`hard` tiers were still elementary word problems (the base cleared them). In
response, those tiers were replaced (the `easy` tier is kept as a near-ceiling
control):

- **`medium`** → genuinely multi-step arithmetic: two-unknown ticket systems, three
  chained percentage changes, combined work rate.
- **`hard`** → **number theory / combinatorics / series** the base cannot shortcut —
  modular exponentiation (`b^e mod 1000`), constrained committee combinations
  (`C(w,j)·C(m,k−j)`), and geometric-series sums. Answers are computed in Python and
  are deliberately **large** (only 1/50 below 100), so the marker-agnostic
  correctness check is not spuriously satisfied by a stray digit in the model's prose.

A **fresh pilot on the hardened bank is pending** — if it clears the gate (baseline
`correctness` < 0.70), the four-job launch and the per-axis leaderboard below follow.

**Expectation still to test (once the bank is hard enough to clear the gate):** each
shape tops the axis it optimizes — `string-match` on `format_rate`,
`code-exec`/`composite` on `correctness`, `autorater` on `explanation_quality` — with
non-overlapping correctness CIs between at least the top correctness reward and the
untuned baseline. A result where `code-exec` also tops `format_rate` is expected and
valid (its reward needs the marker to parse, so it teaches format too).
