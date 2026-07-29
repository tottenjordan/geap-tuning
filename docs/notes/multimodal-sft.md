# Multimodal (image) SFT

How the repo's first **image** tuning example works, and the non-obvious pieces
that aren't recoverable from the code alone. Implemented in
`src/geap_tuning/sft_vision/`, driven by `examples/run_sft_vision.py` and
`notebooks/09_sft_vision.ipynb`. Ports
[jswortz/dental-fine-tune-26](https://github.com/jswortz/dental-fine-tune-26).
Set up 2026-07-29.

## The one thing to remember

**Multimodal SFT is the *same tune call* as text SFT.** Only the records differ —
a `user` turn carries a `fileData` image part. So `sft_vision` has **no tune
module**; it reuses `geap_tuning.sft.tune.launch_sft_job` verbatim. The image
work is all in dataset prep + evaluation.

## Record shape

Each `contents` record pairs an image with the prompt, and the model turn is the
label (built by `data.build_image_records` via `schemas.file_part`):

```json
{"contents": [
  {"role": "user", "parts": [
    {"fileData": {"mimeType": "image/jpeg", "fileUri": "gs://bucket/sft_vision_oral/data/train/caries/caries_001.jpg"}},
    {"text": "Classify the oral disease depicted in this image. Classes: ..."}
  ]},
  {"role": "model", "parts": [{"text": "Dental Caries"}]}
]}
```

The image is referenced by a **`gs://` URI** in the training JSONL (the bytes are
staged to GCS, not inlined). At *eval* time we instead send raw bytes
(`types.Part.from_bytes`) to the tuned endpoint — see the GCS→local trick below.

## Dataset

- Kaggle *Multi-Class Oral Disease Detection Dataset*,
  slug `singh868/multi-class-oral-disease-detection-dataset`, by **Rahul Singh**,
  licensed **CC BY-SA 4.0**. YOLO-structured; 5 classes.
- **Class** is inferred from the **filename prefix** (`calculus`, `cancer`,
  `caries`, `gingivitis`, `ulcer`); **split** from the **path** (`train` /
  `valid|val` / `test`). `LABEL_MAP` maps each prefix → a display label the model
  is trained to emit, and `PROMPT` lists those display labels verbatim.
- Auto-downloaded via **kagglehub** (optional `vision` dependency group). It is
  **lazy-imported** through `importlib.import_module` in `_import_kagglehub`, so
  the package imports and the whole test suite run with the `vision` group *not*
  installed; the missing-dep path raises a `RuntimeError` pointing at
  `uv sync --group vision`.
- Auth: `_configure_kaggle_auth` translates a single `KAGGLE_API_TOKEN` env var
  into what kagglehub wants — if it's the JSON blob `{"username","key"}` it splits
  it into `KAGGLE_USERNAME`/`KAGGLE_KEY`, otherwise it's treated as the key.

## GCS ↔ local mapping (the eval trick)

Images are staged at `{bucket}/{GCS_PREFIX}/data/{split}/{class}/{file}` — note
the literal **`/data/`** segment (`data.DATA_SEGMENT`), which mirrors the local
`out_dir/{split}/{class}/{file}` layout `prepare_dataset` writes. Evaluation reads
each record's `fileData.fileUri`, and `evaluate.resolve_local_path` partitions on
`/data/` and rejoins the tail under the local root — so it can load the **local**
copy's bytes and send them to the endpoint without re-downloading from GCS. Change
`DATA_SEGMENT` and you must change both sides.

## Prediction parsing

The tuned model returns free text; `evaluate.parse_prediction` canonicalizes it to
one of the five display labels by case-insensitive substring match against either
the class key or the display label, falling back to `"unknown"`. Scoring reuses
`sft.evaluate.score_classification` (accuracy + macro-F1).

## Sweep → val-select → test

`run_sft_vision.py` trains **two** configs as separate jobs (reused by display
name), evaluates each on **val** with `run_image_eval`, picks the winner with
`select_best_experiment` (max accuracy, name tie-break), then scores that one on
**test**. Output is a printed comparison table — no plotting deps.

## Cost knobs

Two live tuning jobs + one endpoint call per val/test image. `PER_CLASS` defaults
to a small balanced sample (`train 50 / val 10 / test 10`; the reference used
200/40/40) — raise for fidelity, lower for cost. Jobs are reused by display name
(`geap-sft-vision-{name}`), so re-runs don't re-tune.

## Testing boundary

All of `sft_vision` is pure/offline and unit-tested (`tests/sft_vision/`): synthetic
image trees under `tmp_path`, monkeypatched kagglehub. Every SDK/network call
(`Part.from_bytes`, `generate_content`, `upload_file`, `kagglehub.dataset_download`)
lives in the untested driver/notebook — keep it that way so the package stays
testable without GCP or a Kaggle token.
