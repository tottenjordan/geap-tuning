# GEAP tuning overview

Facts about Gemini Enterprise Agent Platform (GEAP) model tuning, gathered from Google Cloud docs (`docs.cloud.google.com/gemini-enterprise-agent-platform/models/...`) on 2026-07-28. Re-verify model lists and SDK symbols against live docs before relying on them — this surface changes fast. See also [environment](environment.md).

## Tuning methods

| Method | What it does | Notable |
|---|---|---|
| Supervised fine-tuning (SFT) | Teaches a skill from hundreds of labeled JSONL examples | Only option for code models |
| Preference tuning | SFT + human-feedback data for subjective preferences | Gemini 2.5 Flash / Flash-Lite only |
| Tuning checkpoints | Save progress, compare, pick best checkpoint | Toggle "export last checkpoint only" to disable intermediates |
| Continuous tuning | Resume a tuned model/checkpoint with more epochs/data | — |

Modalities: text, image, audio, document (`tune_gemini/{text,image,audio,doc}_tune` docs).

Checkpointing and continuous tuning are **implemented** as cross-cutting demos — see [checkpoints-and-continuous-tuning](checkpoints-and-continuous-tuning.md) for the verified SDK surface (`export_last_checkpoint_only`, per-checkpoint endpoints, default-checkpoint reassignment, and the SFT→RLFT chain).

## Models supporting SFT (as of note date)

Gemini 3.5 Flash, 3.1 Flash-Lite, 2.5 Pro, 2.5 Flash, 2.5 Flash-Lite. Checkpoints + continuous tuning: same list. Preference tuning: only 2.5 Flash / Flash-Lite.

## Job → endpoint flow

1. Stage JSONL train (+ optional validation) dataset in Cloud Storage.
2. Create tuning job — via Cloud console (Agent Platform Studio → "Create tuned model"), Gen AI SDK, Vertex SDK, REST, or Colab Enterprise side panel.
3. Tuning params: number of epochs, adapter size, learning rate multiplier (default 1).
4. Output is a **tuned model endpoint**, not a model name. Call `generate_content` against `tuning_job.tuned_model.endpoint`.
5. Optional: `evaluationConfig` on the job auto-runs Gen AI evaluation after completion (Preview; `us-central1`).

**Thinking models:** turn thinking off / set minimum thinking budget when calling a tuned model. SFT trains it to emit ground truth directly, so the thinking trace is unnecessary and adds cost/latency.

## SDK call shapes

Gen AI SDK:
```python
from google import genai
client = genai.Client(http_options=HttpOptions(api_version="v1"))
job = client.tunings.get(name="projects/.../locations/us-central1/tuningJobs/...")
client.models.generate_content(model=job.tuned_model.endpoint, contents="...")
client.tunings.list()  # yields job objects; use .name
```

Vertex / Agent Platform SDK:
```python
import vertexai
from vertexai.tuning import sft
vertexai.init(project=PROJECT_ID, location="us-central1")
job = sft.SupervisedTuningJob("projects/.../locations/.../tuningJobs/...")
GenerativeModel(job.tuned_model_endpoint_name).generate_content("...")
sft.SupervisedTuningJob.list()  # returns objects, not name strings
```

REST: `GET/POST {REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/{REGION}/tuningJobs`, auth `Bearer $(gcloud auth print-access-token)`.
