# banking77 — a discriminating SFT dataset

Why the repo carries a second SFT dataset, where it comes from, and how it is
sourced. Verified 2026-07-30.

## Why it exists

The demo SFT dataset ([`sft/data.py`](../../src/geap_tuning/sft/data.py)'s
`SUPPORT_TICKETS`) is 65 easy pairs over only **5** intents
(`billing`/`technical`/`account`/`shipping`/`other`). The base
`gemini-2.5-flash-lite` already solves it, so the first SFT DOE
(`examples/run_doe.py`, Experiment `geap-doe-sft`) scores **accuracy = macro_f1 =
1.0 in every grid cell** — a saturated response surface that **cannot
discriminate hyperparameters**. There is no signal to visualize or teach.

**banking77** replaces it for the DOE that has to *show* an improvement: it is a
genuinely hard, roughly-balanced text-classification benchmark with large
base-model headroom, so grid cells separate and tuning visibly beats the untuned
baseline. The saturated `geap-doe-sft` demo is kept on purpose as the "why we
needed a harder dataset" contrast.

## The dataset

- **PolyAI/banking77** — 13,083 customer-service queries labeled with **77
  fine-grained banking intents** (e.g. `card_arrival`, `exchange_rate`,
  `pin_blocked`). Train 10,003 / test 3,080.
- **License: CC-BY-4.0.** Attribution: *Efficient Intent Detection with Dual
  Sentence Encoders*, Casanueva et al., PolyAI 2020.
- Schema: two plain CSVs with a `text,category` header; labels are snake_case.

## Sourcing (no new dependency)

The two CSVs are fetched with **stdlib `urllib.request` + `csv`** and cached
locally — **no extra dependency** (deliberately not `datasets`/`kagglehub`), from:

```
https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv
https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv
```

Loader: [`src/geap_tuning/sft/banking.py`](../../src/geap_tuning/sft/banking.py).
`build_banking_dataset(out_dir, *, csv_dir=None, per_class=..., seed=42)` samples
a balanced subset and writes `train`/`val`/`test` JSONL (standard SFT `contents`
records). Key points:

- **train + val are carved disjoint per label** from the train split; **test** is
  sampled from the held-out test split.
- Every record shares a **`systemInstruction` listing all candidate labels**, so
  both the untuned baseline and the tuned models emit valid labels and the
  before/after comparison is fair. At inference the driver re-supplies the same
  instruction and canonicalizes replies with `parse_banking_prediction` (handles
  casing, trailing punctuation, space↔underscore, substring fallback).
- **Deterministic**: `sample_balanced` uses a fixed-seed `random.Random`, never
  the global RNG — so the sampled subset (and thus the display-name/idempotency
  story) is stable across runs.
- **Offline path**: `--csv-dir` (driver) / `csv_dir=` (loader) point at a
  pre-downloaded copy for air-gapped runs; otherwise CSVs cache under
  `out_dir/raw`. The network download (`download_banking77`) is `# pragma: no
  cover`; all tests exercise the offline `csv_dir` path.
- Cached CSVs + generated JSONL live under `datasets/` (git-ignored).

Demoed by [`examples/run_doe_banking77.py`](../../examples/run_doe_banking77.py) /
[`notebooks/14_doe_banking77.ipynb`](../../notebooks/14_doe_banking77.ipynb) — see
[DOE & visualization](doe-and-visualization.md).
