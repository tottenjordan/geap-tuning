# RLFT reward-shape DOE

**Question:** for the same verifiable-math task, which **reward function** produces
the most accurate tuned model?

Every other DOE in this repo sweeps tuning *hyperparameters*. This one sweeps the
axis unique to RLFT and arguably its most consequential design choice — the
**reward** itself — tuning `gemini-3.5-flash` once per reward shape on the same
dataset and scoring every result on the same held-out split against an **untuned
baseline**, so the before→after lift is explicit.

- **Example:** [`examples/run_doe_rlft_rewards.py`](../../../examples/run_doe_rlft_rewards.py)
- **Notebook:** [`notebooks/15_doe_reward_types.ipynb`](../../../notebooks/15_doe_reward_types.ipynb)
- **Experiment:** `geap-doe-rlft-rewards`
- **Framework mechanics:** [`../../notes/doe-and-visualization.md` → "Sweeping the reward *shape* (RLFT)"](../../notes/doe-and-visualization.md#sweeping-the-reward-shape-rlft)

## The four reward shapes

| Label | `sweep.fixed` payload | What it rewards |
|---|---|---|
| `string-match` | `build_string_match_reward_config()` | the declarative `Answer:\s*-?\d+` **format** (no sandbox) |
| `code-exec` | `build_reward_config()` | **correctness** — ships `geap_tuning.rlft.reward` to the sandbox |
| `autorater` | `build_autorater_reward_config(autorater_model=…)` | **explanation quality**, graded by an LLM judge |
| `composite` | `build_composite_reward_config([(code-exec, 0.8), (autorater, 0.2)])` | correctness (0.8) + quality (0.2) |

The autorater judge needs a **fully-qualified** publisher path
(`projects/<p>/locations/<l>/publishers/google/models/gemini-2.5-flash`); a bare
model name fails with an opaque reward error.

## Why this needs no `doe.py` change

A reward config is a non-scalar object, so it **cannot be a grid axis** (grid
values feed the run slug and Experiments params, which must be scalars). Following
`run_doe_rlft.py`, each reward rides in `sweep.fixed`, where `doe._scalar_params`
keeps it out of the slug, rows, and Experiments params. So each shape becomes its
own **single-run `SweepConfig`** (empty grid → one run), all logged to one shared
Experiment; the driver combines them under **labels it controls**, because every
empty-grid `RunSpec.name` is `"default"` and would otherwise collide.

## How to run

```bash
# run + print the cross-reward table (four RLFT jobs; incurs tuning cost)
uv run python examples/run_doe_rlft_rewards.py

# same, plus write metrics.png into this folder (needs the viz group)
uv run --group viz python examples/run_doe_rlft_rewards.py --plot
```

Reruns are **idempotent** — jobs are reused by display name
(`geap-doe-rew-<shape>-default`), so re-running only launches shapes that don't
already have a finished job.

### Operational gotchas (learned live)

- **Metric semantics — two scores, one generation pass.** The offline eval reports
  both:
  - `accuracy` — **reward-based / marker-gated**: correct *and* in the
    `Answer: <n>` format the reward parser requires (`reward > 0`). This is the
    output contract the reward trains.
  - `content_accuracy` — **marker-agnostic**: does the ground-truth number appear
    anywhere in the reply (`rlft.evaluate.content_correct`)?

  They diverge whenever a model gets the math right but omits the `Answer:` marker.
  Live, **every** run — the untuned baseline and all four tuned shapes — scored
  `accuracy` = 0.0 / `content_accuracy` = 1.0: the models answered every held-out
  problem **correctly in prose** (`"the sum is **55**"`) but none emitted the
  marker at inference. Reporting both columns is what surfaces this: a single
  marker-gated number would read as a flat failure, when in fact the content is
  perfect. (Note this even held for `code-exec`/`composite`, whose reward *is* the
  marker-gated correctness check — so the marker was reinforced in training yet did
  not transfer to the marker-free test prompts.)
- **Tuned-endpoint location.** A tuned Gemini 3.x model is deployed to the `us`
  (or `eu`) **multi-region** endpoint, not the `us-central1` region the job ran
  in. Inference must target the endpoint's own location or it 404s — the example
  uses `config.genai_client_for_endpoint(cfg, endpoint)`. See
  [`../../notes/environment.md`](../../notes/environment.md).
- **Baseline routing.** The untuned baseline is scored through a separate
  `global`-routed inference client (Gemini 3.x base-model inference is
  `global`-only), distinct from the regional tuning client.

## Results

> **Status: complete.** All four RLFT jobs finished (`gemini-3.5-flash`,
> `us-central1`, held-out split of **n = 6**). Scored offline against each tuned
> endpoint plus the untuned baseline; `content_accuracy` is marker-agnostic.

| Reward shape | `accuracy` (marker) | `content_accuracy` | vs. untuned |
|---|---|---|---|
| untuned baseline | 0.000 | 1.000 | — |
| `string-match` | 0.000 | 1.000 | ±0.000 |
| `code-exec` | 0.000 | 1.000 | ±0.000 |
| `autorater` | 0.000 | 1.000 | ±0.000 |
| `composite` | 0.000 | 1.000 | ±0.000 |

![Reward-shape metrics: a grouped bar chart over untuned/string-match/code-exec/autorater/composite showing content_accuracy at 1.0 for every run and marker-gated accuracy at 0.0 for every run.](metrics.png)

### Read this honestly — it's a null result, and that's the finding

The table is **flat**: every run scores identically (content 1.0, marker 0.0), so
`max(accuracy)` picks `string-match = 0.000` only as an arbitrary tie-break. There
is **no best reward shape** here — the DOE did not discriminate. Two reasons, both
worth internalizing before running a reward-shape sweep:

1. **The base model already aces the task.** The untuned `gemini-3.5-flash` baseline
   already scores `content_accuracy` = 1.000 — it solves every held-out problem
   before any tuning. With no content headroom, no reward shape *can* show lift on
   this dataset.
2. **The marker contract never reaches the test.** Marker-gated `accuracy` is 0.000
   everywhere, including `code-exec`/`composite` (whose reward *is* the marker-gated
   check). The `Answer: <n>` format the parser needs isn't emitted on the
   marker-free test prompts — training reinforced it, but it didn't transfer.

**Methodology takeaway:** a reward-shape DOE only measures something when the task
has genuine content headroom for the base model *and* the eval prompt matches the
output contract the reward trains (here, ask for the `Answer:` marker, or score
correctness marker-agnostically as `content_accuracy` does). The small split
(n = 6) compounds both effects. The mechanics this example demonstrates (per-shape
single-run sweeps, driver-owned labels, idempotent reuse, dual-metric scoring) are
the reusable part; the *numbers* are a cautionary tale, not a leaderboard.
