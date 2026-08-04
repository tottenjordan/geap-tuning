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
| SFT | messy text → strict JSON extraction | `sft/extraction.py`, `sft/extraction_eval.py` | field exact-match `accuracy` (+ `json_validity`, `exact_match`) | `examples/run_sft_extraction.py` / `notebooks/17_sft_extraction.ipynb` |
| DPO | concise professional email rewriting | `preference/email.py`, `preference/email_eval.py` | autorater `win_rate` (+ `mean_compression`) | `examples/run_preference_email.py` / `notebooks/18_preference_email.ipynb` |
| RLFT | constrained generation, **graded** reward | `rlft/constrained.py`, `rlft/constraint_reward.py`, `rlft/constraint_eval.py` | constraint `accuracy` (mean graded reward) + `full_satisfaction_rate` | `examples/run_rlft_constrained.py` / `notebooks/19_rlft_constrained.ipynb` |

## Design notes worth keeping

- **SFT extraction headroom.** The gold `quantity` is kept an **int**, and the
  system instruction forbids prose/code fences. Modern bases extract well, so the
  headroom comes from type drift (`"3"` vs `3`) and fence/prose noise; the eval's
  `per_field` breakdown shows where. Comparison is type-insensitive by design
  (`str(v).strip().lower()`), yet the base still loses on validity/format.
- **DPO judge is decoupled.** The A/B autorater prompt is **distinct** from the
  generator's system instruction, blind (A = model under test, B = dispreferred
  reference), and both preference completions carry the **same facts** so it grades
  style, not content. `mean_compression` (rewrite/draft word ratio) is the
  objective backstop. The dataset invariant — preferred word-count < dispreferred
  for **every** triple — is unit-tested and is the concision signal DPO learns.
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

Pending a live authorized run (each incurs one tuning job). Record base→tuned
numbers here after running the three drivers.
