# Knowledge Distillation and Local-Cloud Deployment of a Command-Parsing LM

[![CI](https://github.com/royal-sachan/cloud-command-parser/actions/workflows/ci.yml/badge.svg)](https://github.com/royal-sachan/cloud-command-parser/actions)

**CS3081 Cloud Computing - Mini Project** | Royal Sachan | Roll no. 230101021

| | |
|---|---|
| GitHub repository | https://github.com/royal-sachan/cloud-command-parser |
| Model assets (GitHub Releases) | https://github.com/royal-sachan/cloud-command-parser/releases (`model-v1`, `model-v2`) |
| CI runs | https://github.com/royal-sachan/cloud-command-parser/actions |
| Passing CI run (trained model) | https://github.com/royal-sachan/cloud-command-parser/actions/runs/42460941563 |
| Report / evidence / AI use | `report/report.pdf` / `results/` / `AI_USE.md` |

## What this project does

A small voice-assistant command parser: one English command in, one JSON object out with exactly four keys (`intent`, `date`, `time`, `place`) and 8 possible intents.

```
set an alarm for seven thirty am tomorrow
-> {"intent": "alarm_set", "date": "tomorrow", "time": "seven thirty am", "place": null}
```

A larger **teacher** (Phi-3.5 on Ollama) labels 3,000 commands, the labels are cleaned, and a 70M-parameter **student** (Pythia-70M) is trained on them (sequence-level knowledge distillation, CPU only). The student is served by a FastAPI container on a local Kubernetes cluster (k3d). Model quality is not the goal of the course project; the cloud engineering is.

```
commands_pool.jsonl (3,000, SLURP)  --> SeaweedFS bucket "datasets"
        |
2 parallel k8s workers (SHARD_INDEX 0/1) --> Ollama + phi3.5 (temp 0, seed 42)
        |  datasets/teacher/shard-{0,1}.jsonl
worker/clean.py --> datasets/v1/train.jsonl + train.sha256   (2,616 kept)
        |
training/train.py (Pythia-70M, first 1,500 records) --> bucket "models" v1, v2 + GitHub Releases
        |
FastAPI service (Docker, CPU torch, no weights in image) <-- downloads weights at start-up
        |
k3d / Kubernetes: Deployment (2 replicas) + Service + Traefik Ingress + HPA (1-2)
```

## Results at a glance (measured; see `report/report.pdf` and `results/`)

| Area | Result |
|---|---|
| JSON validity (100 validation commands) | teacher 98%, untrained 0%, students v1, v2, v3 all 100% |
| Full-frame accuracy (intent and all slots correct) | teacher 44%, v1 40%, v2 41%, v3 46% (v3 is an extra experiment, not deployed) |
| Container | image 0.34 GB; 50 of 50 requests OK at 1 CPU / 1 GiB, peak 338 MiB, no OOM; fault injection returns HTTP 500 `invalid_model_output` |
| Rolling update / Pod deletion under load | 0 of 1,338 and 0 of 861 requests failed; rollback returned `X-Model-Version: v1` |
| Cold start | median 8 s over six Pod readings (3 runs) |
| Load balancing (2 replicas) | 301 / 304 requests (49.8% / 50.2%) |
| Final load test (4 users, 2 min) | 460 requests, 0 failed, p50 1,300 ms, p95 1,900 ms, 3.92 req/s |
| Estimated cost (course rate card, not a bill) | 0.56 INR per 1,000 requests |

The validation set has only 100 commands, so differences of a few points between models are not statistically reliable.

## Project status

| Item | Status |
|---|---|
| Tasks 1-5 and 7 (storage, distillation, container, Kubernetes, network, performance) | done, evidence in `results/` |
| CI with the trained model (secret scan, lint, tests, evaluation, quality gate) | passing |
| CI baseline run (untrained model) failing at the quality gate | **not yet recorded** (see Testing and CI) |
| Fresh-clone test of `make up` | not performed |
| `final` tag | create it before submitting |

## Repository layout

| Path | Contents |
|---|---|
| `cluster/k3d.yaml` | k3d cluster config (Ingress and S3 ports) |
| `k8s/` | namespaces, SeaweedFS, worker Job, app Deployment / Service / Ingress, HPA |
| `prompts/teacher_prompt.txt` | teacher prompt |
| `worker/` | labelling worker, `clean.py`, worker Dockerfile |
| `training/` | `train.py`, `config.yaml`, `evaluate.py`, `eval_teacher.py`, `loss.csv` |
| `service/` | FastAPI app, pinned `requirements.txt`, Dockerfile |
| `common/` | shared code (prompt format, post-processing, schema check, S3, teacher client) |
| `network/` | `subnets.md`, `nacl.json`, sample packets |
| `scripts/` | `dev.py` (cluster/deploy), `task4.py`, `perf.py`, `docker_limits.py`, `build_report.py`, `package_submission.py` |
| `tests/`, `.github/workflows/ci.yml` | unit tests, CI pipeline |
| `data/`, `tools/` | course kit (validation set, schema, rate card, `score.py`, Locust file, NACL simulator) plus the cleaned dataset and teacher shards |
| `results/` | logs, score reports, Locust output, command output (evidence) |
| `report/report.pdf`, `AI_USE.md` | report and AI-use disclosure |

## Prerequisites (Windows 10/11, VS Code, PowerShell)

Docker Desktop (WSL2 backend, running), Python 3.11, Git, then:

```powershell
winget install Kubernetes.kubectl
winget install k3d.k3d              # or: choco install k3d
winget install Ollama.Ollama
winget install GitHub.cli           # for releases
choco install make                  # optional; otherwise run the Python commands from the Makefile
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

Let the worker Pods reach Ollama on the laptop: `setx OLLAMA_HOST 0.0.0.0`, quit and restart Ollama, then `ollama pull phi3.5`. The workers reach it through `host.docker.internal`.

## Quick start

Needs the GitHub Releases `model-v1` and `model-v2` to exist. (Documented flow; a fresh-clone run has not been tested yet.)

```powershell
make up        # cluster + SeaweedFS + buckets + data/model restore + image build + deploy
curl.exe -X POST http://localhost:8081/parse -H "Content-Type: application/json" -d "{\"text\": \"what is the weather in london\"}"
make down      # delete the cluster
```

Expected answer: `{"intent":"weather_query","date":null,"time":null,"place":"london"}` with response headers `X-Model-Version` and `X-Served-By`. The host port (8081 here) is set in `cluster/k3d.yaml`; the helper scripts must use the same port.

## Service API

| Endpoint | Behaviour |
|---|---|
| `POST /parse` body `{"text": "..."}` | 200 and the four-key JSON object; headers `X-Model-Version`, `X-Served-By` (Pod name). Invalid model output gives 500 `{"error": "invalid_model_output"}`, never broken JSON with 200 |
| `GET /healthz` | 200 while the process is alive |
| `GET /readyz` | 503 until the model is loaded, then 200 |

Configuration (environment): `MODEL_URI`, `MODEL_VERSION`, `S3_ENDPOINT`, `MAX_NEW_TOKENS`; S3 credentials come from a Kubernetes Secret; `FAULT_INJECT=1` replaces the model output with `not json` to test validation.

## Full workflow (how the evidence in `results/` was produced)

| Task | Commands |
|---|---|
| **1** Storage + labelling | `make infra` -> `make label` (hours; follow with `kubectl -n pipeline logs -f -l job-name=teacher-labelling --prefix`) -> `make clean-data` |
| **2** Distillation | `make train SUBSET=50` (smoke test) -> `make train` -> `make train-v2` -> `make upload-model VERSION=v1` / `v2` -> `make teacher-eval` -> `make eval-all` -> `gh release create model-v1 dist/model-v1.zip` / `model-v2 dist/model-v2.zip` |
| **3** Container | `make build` -> `python scripts/docker_limits.py` (50 requests at 1 CPU / 1 GiB, peak memory) -> `python scripts/docker_limits.py fault` (expects HTTP 500) |
| **4** Kubernetes | `python scripts/dev.py app` -> `python scripts/task4.py` with `endpoints`, `rolling`, `recovery`, `coldstart`, `lb`, `hpa` |
| **5** Network | `python tools/nacl_sim.py network/nacl.json network/packets_sample.jsonl --check` (working in `network/subnets.md`) |
| **6** CI/CD | push; Actions -> `ci` -> Run workflow with `model=baseline` (must fail at the quality gate) |
| **7** Performance | `python scripts/task4.py lb` -> `python scripts/perf.py` -> `python scripts/build_report.py` |

No `make`? Every target is one Python command; see the `Makefile`.

Checksums: `Get-FileHash dist\model-v1.zip -Algorithm SHA256` must equal the hash in `models_out/v1/model.sha256`; `data/train.sha256` is the hash of `data/train.jsonl`.

## Testing and CI

`make lint` (ruff) and `make test` (pytest: cleaner, post-processing, service endpoints, fault injection, NACL file). GitHub Actions runs, in order: lint -> unit tests -> model evaluation (downloads `model-v1.zip` from the Release and runs `score.py` on `validation.jsonl`) -> quality gate (`score.py --min-validity 95`), plus a `gitleaks` secret scan.

To demonstrate the gate failing, run the workflow manually (Actions -> ci -> Run workflow) with `model=baseline`; it evaluates the untrained Pythia-70M, which produces 0% valid JSON. Then put the run URL in `BASELINE_RUN_URL` at the top of `scripts/build_report.py` and rebuild the report.

## Design notes

- **Reproducible data:** shards are split by `line_number % SHARD_COUNT`; `clean.py` is deterministic and writes LF-only bytes, so the SHA-256 is identical on Windows and Linux. Teacher replies are stored exactly as generated, and workers save progress every 20 records so they can resume.
- **Distillation method:** sequence-level (cross-entropy on the teacher's JSON text, loss on the response only). Logit-based distillation is not possible because Phi-3.5 and Pythia use different tokenizers and only text was stored.
- **No secrets in Git:** S3 keys are generated at `make up` into the git-ignored `.secrets/` folder and injected as Kubernetes Secrets.
- **No weights in Git or in the image:** the service downloads `MODEL_URI` from SeaweedFS at start-up; `/readyz` returns 503 until the model is loaded.
- **Single-node limitation:** the cluster runs on one laptop, so it shows self-healing and rolling updates but not fault tolerance across machines.

## Troubleshooting (Windows)

| Problem | Fix |
|---|---|
| `make` is not recognized | `choco install make`, reopen the terminal, or run the Python commands directly |
| Pods cannot reach Ollama | set `OLLAMA_HOST=0.0.0.0`, restart Ollama, and check the Ollama URL in `k8s/worker-job.yaml` |
| `SeaweedFS S3 not reachable on localhost:8333` | wait for the pod (`kubectl -n data get pods`), confirm port 8333 is free and Docker Desktop is running |
| Port already in use | change the host port in `cluster/k3d.yaml`, recreate the cluster, and use the same port in `scripts/task4.py` and `scripts/perf.py` |
| Labelling is slow | CPU-only Ollama is slow; the workers resume from saved progress, so a restart loses nothing |
| Model container fails to load | check `kubectl -n app logs <pod>`; the service needs the pinned dependencies in `service/requirements.txt` (including `accelerate`) |

## Submission checklist

1. `report/report.pdf` (at most 8 pages) and the `results/` evidence.
2. Scanned handwritten reflection at `reflection/230101021.pdf`.
3. Model files only as GitHub Release assets (never committed); no `.secrets/` in the repo.
4. Tag the final commit: `git tag final` then `git push --tags`.
5. `python scripts/package_submission.py` -> `submission/230101021.zip` (report, reflection, source, this README); upload only that zip.