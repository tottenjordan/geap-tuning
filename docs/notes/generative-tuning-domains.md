# Generative tuning demos in fresh domains (SFT / DPO / RLFT)

Three self-contained before→after demos, one per GEAP tuning service, in problem
domains the repo had **not** shown. Each scores the untuned base and the tuned
endpoint on the same held-out split and prints the metric lift. They reuse the
established recipe (pure dataset module → `schemas.*_example` + `write_jsonl` →
stage to GCS → score base → tune once, reuse-by-display-name → score tuned) and
all shared plumbing (`config`, `gcs`, `inference`, `jobs`, the three launchers).
No new dependencies, no `doe.py` changes — these are single-tune demos, not sweeps.

## Why these domains

The prior demos clustered: SFT was **only classification** (support-intent,
banking77, oral-disease images); DPO was **only** support-reply warmth (no
recorded lift); RLFT was **only** verifiable math, which produced two documented
null results (see [DOE reward ranking](../doe/rlft-reward-ranking/README.md)). The
new domains widen the story and each targets an objective metric with real
headroom.

| Service | New domain | Module(s) | Headline metric | Driver / notebook |
|---|---|---|---|---|
| SFT | messy text → strict JSON extraction, applying a **house normalization standard the base cannot guess** | `sft/extraction.py`, `sft/extraction_eval.py` | field exact-match `accuracy` (+ `per_field`, `json_validity`) | `examples/run_sft_extraction.py` / `notebooks/17_sft_extraction.ipynb` |
| DPO | concise professional email rewriting | `preference/email.py`, `preference/email_eval.py` | **objective `mean_compression`** (+ `compression_win_rate`); subjective `win_rate` secondary | `examples/run_preference_email.py` / `notebooks/18_preference_email.ipynb` |
| RLFT | constrained generation, **graded** reward | `rlft/constrained.py`, `rlft/constraint_reward.py`, `rlft/constraint_eval.py` | constraint `accuracy` (mean graded reward) + `full_satisfaction_rate` | `examples/run_rlft_constrained.py` / `notebooks/19_rlft_constrained.ipynb` |

## Design notes worth keeping

- **SFT extraction headroom = a convention the base cannot guess.** Plain field
  extraction saturates a modern base (it scored `accuracy=1.000` in a live run), so
  the task was redesigned: the gold object applies a **house normalization standard**
  the raw line does not reveal — strip the `ord-` prefix and upper-case the id
  (`ord-g8850`→`G8850`), expand the city abbreviation (`PHL`→`Philadelphia`), map the
  urgency word to an arbitrary P-code (`urgent`→`P0`), and integerize spelled-out
  counts (`a dozen`→`12`). `SYSTEM_INSTRUCTION` *signals a standard applies* but
  never spells out the arbitrary mapping — SFT must learn it from labels. Comparison
  is type-insensitive (`str(v).strip().lower()`); the `per_field` breakdown localizes
  exactly which conventions the base misses. **Key finding: not all conventions are
  equally learnable — see Measured lifts.**
- **DPO headline is objective concision, not the subjective judge.** A strong base
  already out-writes hand-authored gold on a blind A/B judge (it wins 86.7% while
  *expanding* the draft), so the subjective `win_rate` saturates and is kept only as
  a secondary "what DPO doesn't move" signal. The honest headline is the objective
  axis the preference pairs actually train: `mean_compression` (rewrite/draft word
  ratio; <1 is shorter) plus `compression_win_rate` (fraction of drafts where the
  tuned rewrite is strictly shorter than the base rewrite, a binomial rate that takes
  `bootstrap_ci`). The judge prompt is still **distinct** from the generator's system
  instruction, blind, and both completions carry the **same facts**. The dataset
  invariant — preferred word-count < dispreferred for **every** triple — is
  unit-tested and is the concision signal DPO learns. The pilot gate refuses to spend
  unless base `mean_compression >= 0.9` (real concision headroom).
- **RLFT graded reward answers both prior nulls.** The reward is the *fraction* of
  independently-checked components satisfied (each required keyword and each
  forbidden word is its own component; the word-count band and sentence-count band
  are one component each). Fractional credit gives the reward **variance** even for
  a mediocre rollout (fixes the zero-gradient null), and requiring four constraint
  types at once leaves **headroom** even for a strong base (fixes the saturation
  null). Contrast the math sweep, where a binary `Answer:` reward could not teach a
  format the base emitted ≈0% of the time — RL cannot bootstrap a from-scratch
  output format without an SFT warm-start.
- **Reward ships verbatim.** `constraint_reward.py` is stdlib-only, pure, and
  self-contained (a unit test parses its AST to assert it imports only `re`/
  `typing`), because `build_reward_config("constraint_satisfaction", module=...)`
  sends its source to the GEAP code-execution sandbox via `inspect.getsource`. The
  new `module=` kwarg (default stays `rlft.reward`) is what lets any reward module
  ride the same launcher. `component_breakdown` is the single source of truth for
  both the sandbox reward and the offline `constraint_eval`.
- **RLFT base + region.** `gemini-3.5-flash` is the only RLFT-supported base; the
  tuning client stays **regional** (the `global` endpoint serves Gemini 3.x
  inference but not tuning), while the untuned baseline runs against the global
  inference endpoint (`genai_client(cfg, base_model=...)`) and the tuned endpoint
  is scored via `genai_client_for_endpoint`.
- **Pilot gate (RLFT only).** `run_rlft_constrained.py --pilot-only` scores the
  untuned base and stops for free; it refuses to launch unless base constraint
  `accuracy < SAT_CEILING` (0.85). `--force` overrides. This is the cheap headroom
  check that the two prior math nulls motivated. `bootstrap_ci` reports a 95% CI on
  the full-satisfaction rate.

## RLFT pilot-gate finding (exact counts vs. bands)

Ran `run_rlft_constrained.py --pilot-only` (free — scores the untuned base only)
against `gemini-3.5-flash` (`us-central1`, n=30 held out):

- **Band-based constraints saturate.** The first bank used count *bands* ("between
  40 and 90 words", "between 3 and 6 sentences"). The base scored `accuracy=0.992`,
  `full_satisfaction_rate=0.933` — keywords/forbidden 1.000, word/sentence bands
  ~0.97. The gate correctly refused to spend (no headroom).
- **Exact counts reopen headroom.** Switching to exact targets ("exactly 55 words,
  exactly 4 sentences", encoded as `min == max` so the reward is unchanged) drops
  the base to `accuracy=0.823`, `full_satisfaction_rate=0.000` — and the headroom
  is fully localized: `word_count 0.000 (0/30)` while keywords 0.986, forbidden
  1.000, sentence_count 1.000. The base can hit an exact *sentence* count but never
  an exact *word* count. Keeping keywords/forbidden modest (2–3 / 1–2 per prompt)
  keeps the easy components from masking the hard one in the micro-average.
- **Structural caveat vs. the marker null.** Exact word count has a ~0% base rate,
  echoing the prior marker null — but unlike a from-scratch token format, the base
  always emits *some* word count, so across `samples_per_prompt` rollouts some land
  on the target by natural variation, giving RLFT an advantage signal to climb.
  Whether that yields a measurable word-count lift is the open question a live run
  answers.

## Measured lifts

### RLFT constrained generation — a *third*, predicted null (ran live)

`run_rlft_constrained.py` ran end-to-end on `gemini-3.5-flash` (`us-central1`,
tuning job `SUCCEEDED` in ~53 min, n=30 held out). The pilot gate passed
(base `accuracy=0.823 < 0.85`, headroom fully localized to exact word count),
the job tuned, and the tuned endpoint was scored:

| Metric | Base | Tuned | Δ |
|---|---|---|---|
| `accuracy` (mean graded reward) | 0.823 | 0.828 | +0.006 |
| `full_satisfaction_rate` | 0.000 (CI [0,0]) | 0.000 (CI [0,0]) | — |
| keywords | 0.986 | 1.000 | +0.014 |
| forbidden | 1.000 | 1.000 | — |
| **word_count** (the headroom axis) | 0.000 (0/30) | 0.000 (0/30) | **—** |
| sentence_count | 1.000 | 1.000 | — |

**The one moving component was `keywords` (0.986 → 1.000)** — a behavior the base
already emitted ~99% of the time, which RL amplified to 100%. The exact-word-count
axis, the only real headroom, stayed at 0/30 before *and* after. This is the
structural caveat above, confirmed: an exact word count has a ~0% base rate, so
nearly every rollout earns the same word-count reward → no advantage signal → no
gradient. It is the same mechanism as the prior `Answer:`-marker null (see
[DOE reward ranking](../doe/rlft-reward-ranking/README.md)): **RL amplifies
behaviors the base sometimes produces; it cannot bootstrap a near-never behavior
without an SFT warm-start.** The graded reward *did* deliver the promised
variance (keywords moved, and the base's graded `accuracy` was 0.82 not 0.00), so
the reward design is sound — the null is a genuine property of reinforcement
tuning, and the pilot gate correctly identified real (but, as it turns out,
un-RL-reachable) headroom.

### SFT extraction — SFT learns *rule-based* conventions, resists an *arbitrary relabel* (ran live)

The v1 plain-extraction task saturated (`accuracy=1.000`), so it was redesigned to
teach a house normalization standard (above) and re-gated with a pilot. Two live
tunes on `gemini-2.5-flash` (`us-central1`, n=30 held out):

- **v2 (`epochs=2`, `adapter=8`) — no lift.** base `accuracy=0.727` → tuned `0.727`.
  It learned the transforms the base half-knew but not the arbitrary map. Diagnosed
  as under-training and retuned.
- **v3 (`epochs=6`, `adapter=16`) — a modest, telling lift.**

| Field | Convention kind | Base | Tuned (v3) |
|---|---|---|---|
| `accuracy` (micro) | — | 0.727 | **0.787** (+0.060) |
| `order_id` | rule (strip `ord-` + upper-case) | 0.667 | **1.000** |
| `city` | lookup the base already half-knows (`PHL`→Philadelphia) | 0.967 | 0.967 |
| `quantity` | rule (spelled-out → int) | 1.000 | 0.967 |
| **`priority`** | **fully-arbitrary relabel (`urgent`→`P0`)** | **0.000** | **0.000** |

**The entire +0.060 lift is `order_id` going to perfect; `priority` never moved —
even with 3× epochs and 2× adapter.** A direct endpoint probe confirms the tuned
model applies every rule-based transform (`ord-g8850`→`G8850`, `PHL`→`Philadelphia`,
`a dozen`→`12`) but **never emits a single P-code** — it echoes the raw urgency word
(`high`, `Low`, `normal`), and in one case even relabels `urgent`→`high`. So this is
not simple under-fit: within reachable LoRA hyperparameters, SFT readily learns
deterministic **string-transform** conventions that align with the model's
generalization, but **resists an arbitrary categorical relabel that fights a strong
semantic prior** (the base "knows" `urgent` is a priority word and refuses to overwrite
it with an opaque code). A parallel to the RLFT nulls from the other side: RLFT can't
teach a from-scratch behavior for lack of gradient signal; SFT *has* the signal here
yet the prior still dominates one field. The task design is sound (headroom was real
and localized); the honest lesson is *which* conventions LoRA-SFT will and won't teach.
The v2-vs-v3 sweep and the per-field chart are written up as a DOE:
[`../doe/sft-extraction-convention/README.md`](../doe/sft-extraction-convention/README.md).

### DPO email — a modest objective concision gain; subjective judge stays flat (ran live)

`run_preference_email.py` ran on `gemini-2.5-flash` (`us-central1`, `epochs=3`,
`beta=0.2`, n=15 held out). The pilot gate passed (base `mean_compression=1.12 ≥ 0.9`
— the base *expands* drafts, real concision headroom):

| Metric | Base | Tuned | Note |
|---|---|---|---|
| `mean_compression` (headline; lower = shorter) | 1.13 | **1.04** | base expands +13% → tuned +4% |
| `compression_win_rate` (tuned strictly shorter than base) | — | 0.533 (8/15) | CI [0.267, 0.800] — not distinguishable from 0.5 at n=15 |
| subjective judge `win_rate` (secondary) | — | 0.333 | judge still prefers the verbose base |

DPO moved the exact axis it trains — `mean_compression` fell from +13% to +4% — but
weakly: the per-draft "strictly shorter" rate is a coin flip at this sample size and
the subjective judge, which a strong base already saturates, does not reward the
extra concision. An honest before→after: the trained objective improves directionally,
while the subjective axis a strong base dominates does not — the same theme as the
SFT and RLFT results, that modern bases leave thin, uneven headroom.
