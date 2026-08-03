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

- **Eval must replay the training framing (the biggest one).** These RLFT records
  are trained under a **system instruction** ("…end with the final answer on its
  own line as 'Answer: `<number>`'") that carries the very output contract the
  reward scores. The first version of this eval sent only the bare user question
  and **dropped that system instruction** — so the model answered in free prose,
  never emitted the marker, and scored `accuracy` = 0.000 across *every* shape.
  That was a **measurement artifact, not a training outcome**: an A/B probe of the
  same `code-exec` endpoint scored 0.000 on bare prompts and 1.000 once the system
  instruction was restored. The fix (`rlft/evaluate.py`) threads each record's
  `systemInstruction` through `run_rlft_eval` → `generate_fn`, keeping inference
  faithful to training. **Lesson: an RLFT eval must reproduce the training prompt
  exactly — same system instruction, same framing — or it measures the mismatch,
  not the model.**
- **Metric semantics — two scores, one generation pass.** The offline eval reports
  both:
  - `accuracy` — **reward-based / marker-gated**: correct *and* in the
    `Answer: <n>` format the reward parser requires (`reward > 0`). This is the
    output contract the reward trains.
  - `content_accuracy` — **marker-agnostic**: does the ground-truth number appear
    anywhere in the reply (`rlft.evaluate.content_correct`)?

  They diverge whenever a model gets the math right but omits the `Answer:` marker
  — exactly the failure mode the dropped-system-instruction bug above produced
  (content 1.0, marker 0.0). Once the eval replays the system instruction, both
  land at 1.0 here, so reporting both columns is what let us catch the artifact:
  a single marker-gated number would have read as a flat failure while the content
  was in fact perfect.
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
> endpoint plus the untuned baseline, **replaying each record's training system
> instruction** (see the gotcha above); `content_accuracy` is marker-agnostic.

| Reward shape | `accuracy` (marker) | `content_accuracy` | vs. untuned |
|---|---|---|---|
| untuned baseline | 1.000 | 1.000 | — |
| `string-match` | 1.000 | 1.000 | ±0.000 |
| `code-exec` | 1.000 | 1.000 | ±0.000 |
| `autorater` | 1.000 | 1.000 | ±0.000 |
| `composite` | 1.000 | 1.000 | ±0.000 |

![Reward-shape metrics: a grouped bar chart over untuned/string-match/code-exec/autorater/composite showing both marker-gated accuracy and content_accuracy at 1.0 for every run.](metrics.png)

### Read this honestly — it's a null result, and that's the finding

The table is **flat**: every run scores 1.000 on both metrics, so `max(accuracy)`
picks `string-match` only as an arbitrary tie-break. There is **no best reward
shape** here — the DOE did not discriminate. The reason is worth internalizing
before running a reward-shape sweep:

- **The base model already aces the task — content *and* format.** Given the same
  system instruction it was tuned under, the untuned `gemini-3.5-flash` baseline
  already scores `accuracy` = 1.000 *and* `content_accuracy` = 1.000: it solves
  every held-out problem **and** emits the `Answer: <n>` marker before any tuning.
  With no headroom on either axis, no reward shape *can* show lift on this dataset.

> **Correcting an earlier writeup.** An initial version of this doc reported
> `accuracy` = 0.000 for every run and concluded the marker contract "didn't
> transfer to the test." **That was wrong** — it was the dropped-system-instruction
> measurement bug (see the first gotcha), not a training outcome. Once the eval
> replays the system instruction, the marker is emitted perfectly everywhere. The
> honest finding is a null result for a *different* reason: the base model already
> saturates the task.

**Methodology takeaway:** a reward-shape DOE only measures something when (a) the
task has genuine headroom for the base model, and (b) the **eval reproduces the
training framing** — same system instruction — so you measure the model, not a
prompt mismatch. The small split (n = 6) compounds the ceiling effect. The
mechanics this example demonstrates (per-shape single-run sweeps, driver-owned
labels, idempotent reuse, dual-metric scoring, **train/eval prompt parity**) are
the reusable part; the *numbers* are a cautionary tale, not a leaderboard.
