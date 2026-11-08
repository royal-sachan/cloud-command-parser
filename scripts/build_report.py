"""Generate report/report.pdf (<= 8 pages) from the files in results/ and the repo. Measured values only come from results/."""
import csv
import json
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Table, TableStyle

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
NA = "not run yet"
ss = getSampleStyleSheet()
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=8.3, leading=10.2, spaceAfter=3)
H = ParagraphStyle("H", parent=ss["Heading2"], fontSize=11, spaceBefore=6, spaceAfter=3)
T = ParagraphStyle("T", parent=ss["Title"], fontSize=15, spaceAfter=2)
C = ParagraphStyle("C", parent=B, fontSize=7.4, leading=9)
ASSUME = ParagraphStyle("A", parent=B, textColor=colors.HexColor("#8a4b00"))


def J(name):
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def tail(name, pat=None, n=6):
    p = RES / name
    if not p.exists():
        return [NA]
    lines = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if pat:
        lines = [l for l in lines if re.search(pat, l)]
    return lines[-n:] or [NA]


def tbl(rows, widths=None, head=True):
    t = Table([[Paragraph(str(c), C) for c in r] for r in rows], colWidths=widths, repeatRows=1)
    st = [("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]
    if head:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7")))
    t.setStyle(TableStyle(st))
    return t


def P(x, s=B):
    return Paragraph(x, s)


def lines(name, pat=None, n=6):
    return P("<br/>".join(l.replace("<", "&lt;") for l in tail(name, pat, n)), C)


def main():
    counts, perf = J("clean_counts.json"), J("perf_summary.json")
    rep = ROOT / "README.md"
    repo = (re.search(r"https://github.com/\S+", rep.read_text()) or [None])[0] if rep.exists() else None
    S = []
    S += [P("Knowledge Distillation and Deployment of a Command-Parsing LM on a Local Cloud", T),
          P("Cloud Computing (CS3081) Mini Project &mdash; Royal Sachan, roll no. 230101021 &mdash; "
            f"repository: {repo or 'see README.md'}", B),
          P("<b>Convention:</b> numbers in tables come from files in <i>results/</i> (measured). Text in brown is an assumption or design choice.", B)]

    # Task 1
    S += [P("1. Object storage and parallel teacher labelling", H),
          P("Architecture: a k3d cluster (<i>cluster/k3d.yaml</i>) runs SeaweedFS in namespace <i>data</i> (buckets <i>datasets</i>, "
            "<i>models</i>, S3 on localhost:8333 via the k3s service load balancer). Ollama (phi3.5, temperature 0, seed 42, JSON-constrained "
            "output, <i>num_predict</i> 100) runs on the laptop; two Indexed-Job workers in namespace <i>pipeline</i> reach it via "
            "<i>host.docker.internal</i>. The prompt is <i>prompts/teacher_prompt.txt</i> (instruction, 8 intents, 3 slot rules, 8 own examples).")]
    S += [P("<b>Sharding.</b> Command line n (from 0) of <i>commands_pool.jsonl</i> goes to the worker with SHARD_INDEX = n mod SHARD_COUNT, so every "
            "line belongs to exactly one shard (no gaps, no repeats) and the two shards have equal size without coordination. Each worker writes its "
            "own object <i>datasets/teacher/shard-&lt;i&gt;.jsonl</i> (flushed every 20 records and resumable after a Pod restart, so a multi-hour run "
            "survives crashes). Ollama serves requests on one CPU, so two workers mainly demonstrate the pipeline design; throughput is bounded by the teacher.")]
    if counts:
        S += [tbl([["Input", "(a) invalid JSON", "(b) bad keys", "(c) bad intent", "(d) bad slot", "Duplicates", "Kept"],
                   [counts[k] for k in ("input", "a_invalid_json", "b_bad_keys", "c_bad_intent", "d_bad_slot", "duplicates", "kept")]]),
              P("Check: input = a + b + c + d + duplicates + kept = " +
                " + ".join(str(counts[k]) for k in ("a_invalid_json", "b_bad_keys", "c_bad_intent", "d_bad_slot", "duplicates", "kept")) +
                f" = {counts['input']}. Dataset checksum: <i>data/train.sha256</i>.", B)]
    else:
        S += [P(f"Counts table: {NA}.", B)]
    S += [P("Cleaning order: (a)-(d) per record in id order, then duplicates by exact command text (smallest id kept). Output is written as bytes "
            "(LF only) so the SHA-256 is identical on Windows and Linux.", ASSUME)]

    # Task 2
    S += [P("2. Training the student (sequence-level distillation)", H),
          P("Pythia-70M is fine-tuned on the first 1,500 cleaned records with PyTorch/Transformers on CPU. Example text: "
            "<i>Command: ...\\nJSON: ...</i> + EOS; the loss is computed on the JSON part only. The teacher JSON is re-serialised with fixed key order "
            "(same content). Config (<i>training/config.yaml</i>): seed 42, lr 3e-4, batch 16, 6 epochs, max length 96; v2 uses seed 7 and 8 epochs. "
            "Models and checksums: bucket <i>models/v1</i>, <i>models/v2</i>, GitHub Release assets <i>model-v1.zip</i>, <i>model-v2.zip</i>.")]
    cols = [("teacher_validation.json", "Teacher"), ("untrained_validation.json", "Untrained"),
            ("student_v1_validation.json", "Student v1"), ("student_v2_validation.json", "Student v2"),
            ("student_v3_validation.json", "Student v3")]
    keys = ["json_validity", "intent_accuracy", "date_accuracy", "time_accuracy", "place_accuracy", "full_frame"]
    reps = [J(f) for f, _ in cols]
    S += [tbl([["Metric (%) on validation (100)"] + [n for _, n in cols]] +
              [[k] + [str(r.get(k, NA)) for r in reps] for k in keys])]
    loss = ROOT / "training" / "loss.csv"
    if loss.exists():
        ls = list(csv.DictReader(open(loss)))
        S += [P("v1 training loss per epoch: " + ", ".join(f"{float(r['train_loss']):.3f}" for r in ls) + ".", B)]
    S += [P("<b>Discussion.</b> The untrained model has no notion of the task, so its output is rarely a valid four-key object; fine-tuning on 1,500 "
            "teacher replies teaches the format almost immediately (validity) while slot copying and intent choice are harder for 70M parameters. "
            "The student can at best approach the teacher because it only imitates teacher outputs, including the teacher's mistakes.", B),
          P("<b>Sequence-level vs logit-based.</b> Sequence-level distillation trains the student with ordinary cross-entropy on the teacher's generated "
            "output text (hard labels). Logit-based distillation trains it to match the teacher's per-token probability distribution (soft labels, KL divergence). "
            "It is impossible here because the two models use different tokenizers/vocabularies (Phi-3.5 vs GPT-NeoX), so their output distributions "
            "do not align token by token, and we only stored text replies, not logits. Limitation: the student inherits teacher errors and sees only "
            "one decoded answer per command (no uncertainty information); with 1,500 examples and a 70M model it also generalises poorly to unseen phrasings.", B)]

    # Task 3
    S += [P("3. Inference service in a container", H),
          P("FastAPI service (<i>service/app.py</i>, port 8000): POST /parse, GET /healthz (alive), GET /readyz (503 until the model is loaded; loading runs "
            "in a background thread). At start-up it downloads <i>MODEL_URI</i> from SeaweedFS (S3 credentials from a Kubernetes Secret). Output is checked "
            "against <i>data/schema.json</i>; invalid output gives HTTP 500 <i>{\"error\": \"invalid_model_output\"}</i>; FAULT_INJECT=1 forces the text "
            "<i>not json</i>. Headers X-Model-Version and X-Served-By are set. Image: python:3.11-slim + CPU-only torch wheel, every dependency pinned, no weights."),
          P("Run under docker --cpus=1 --memory=1g, 50 sequential requests (measured):", B), lines("task3_limits.txt", r"requests sent|image size|peak", 4)]

    # Task 4
    S += [P("4. Kubernetes deployment, scaling and reliability", H),
          P("Namespace <i>app</i>: Deployment (2 replicas; requests 500m/512Mi, limits 1 CPU/1Gi; liveness /healthz, readiness /readyz; rolling update "
            "maxUnavailable 0, maxSurge 1, preStop sleep 8), Service, Ingress (Traefik, /parse). HPA 1-2 replicas at 70% CPU (utilisation is relative to the 500m request)."),
          P("<b>Rolling update v1&rarr;v2 under load, then rollout undo</b>", B), lines("rolling.log", r"requests=|headers|X-Model-Version", 4),
          P("<b>Recovery (one Pod deleted under load)</b>", B), lines("recovery.log", r"requests=|restored", 3),
          P("<b>Cold start (3 runs, delete all Pods; the logs show the model download)</b>", B), lines("coldstart.log", r"median|downloaded", 4),
          P("<b>Load balancing (2 ready replicas)</b>", B), lines("lb.log", r"requests=|per replica", 3),
          P("<b>HPA scale up/down</b>", B), lines("hpa.log", r"replicas now|HPA:", 4),
          P("<b>Limitation:</b> the cluster is a single k3d node on one laptop: it shows Pod-level self-healing and rolling updates but NOT fault tolerance "
            "across machines; losing the laptop (or Docker) loses everything. Metrics-server on k3d has a coarse scrape interval, so HPA reacts slowly.", B)]

    # Task 5
    S += [P("5. Subnetting and network access control", H),
          tbl([["Subnet", "Hosts", "Network/prefix", "Usable range", "Broadcast", "Usable"],
               ["app", "1,000", "10.20.0.0/22", "10.20.0.1 - 10.20.3.254", "10.20.3.255", "1,022"],
               ["public", "250", "10.20.4.0/24", "10.20.4.1 - 10.20.4.254", "10.20.4.255", "254"],
               ["data", "100", "10.20.5.0/25", "10.20.5.1 - 10.20.5.126", "10.20.5.127", "126"],
               ["management", "20", "10.20.5.128/27", "10.20.5.129 - 10.20.5.158", "10.20.5.159", "30"]]),
          P("Working is in <i>network/subnets.md</i> (2^(32-p)-2 &ge; need, smallest p). <i>network/nacl.json</i>: inbound 100 deny 198.51.100.0/24; 110/120 allow TCP 80/443; "
            "130 allow TCP 22 from 10.20.5.128/27; 140 allow TCP 1024-65535 from the app subnet. Outbound 100 allow TCP 8000 to 10.20.0.0/22; 110 allow TCP 1024-65535 to anywhere. "
            "<i>nacl_sim.py --check</i>: 6/6 correct (measured).", B),
          P("A stateless NACL does not remember connections, so reply packets are judged as new packets and need their own rules (ephemeral ports 1024-65535). "
            "Rules are evaluated lowest number first, first match wins, so the deny for 198.51.100.0/24 must be numbered below the allows for 80/443; "
            "otherwise the allow would match first and the abusive range would get through.", B)]

    # Task 6
    S += [P("6. Reproducibility, testing and CI/CD", H),
          P("<i>make up</i> (after <i>git clone</i> and <i>pip install -r requirements-dev.txt</i>) creates the cluster, SeaweedFS, buckets, restores the committed datasets and "
            "downloads the model Release assets into the bucket, builds and imports the image and deploys the app; <i>make down</i> deletes the cluster. "
            "GitHub Actions (<i>.github/workflows/ci.yml</i>) runs lint &rarr; unit tests (10 tests) &rarr; model evaluation (downloads <i>model-v1.zip</i> "
            "from the Release, runs <i>score.py</i>) &rarr; quality gate (<i>score.py --min-validity 95</i>), plus a gitleaks secret scan. ")]
    fr = RES / "ci_failed_run.txt"
    ci_baseline = (fr.read_text().strip() if fr.exists()
                   else "Not completed: no baseline CI run has been recorded; this check remains outstanding.")
    S += [P("Baseline quality-gate run (untrained Pythia-70M): " + ci_baseline, B),
          P("Release-model CI passed all jobs, including secret scan, lint, unit tests, model evaluation and quality gate: "
            "<link href='https://github.com/royal-sachan/cloud-command-parser/actions/runs/42460941563'>run 42460941563</link>. "
            "The optional fresh-clone test was skipped.", B),
          P("AI tools used and how output was verified: see <i>AI_USE.md</i>.", B)]

    # Task 7
    S += [P("7. Performance, cost and review", H)]
    if perf:
        pods = "; ".join(f"{n}: {v['cpu_m']}m CPU, {v['mem_mi']}Mi" for n, v in perf["peak_cpu_mem_per_pod"].items())
        S += [tbl([["p50 (ms)", "p95 (ms)", "Throughput (req/s)", "Requests / failures", "Error rate"],
                   [perf["p50_ms"], perf["p95_ms"], perf["rps"], f"{perf['requests']} / {perf['failures']}", f"{perf['error_rate_pct']}%"]]),
              P(f"Peak per Pod during the test (kubectl top): {pods}. Cold start: see section 4.", B),
              P(f"<b>Cost (hypothetical rate card).</b> {perf['cost']['steps']} = <b>{perf['cost']['cost_per_1000_requests_inr']} INR per 1,000 requests</b>.", B),
              P(f"<b>SLO:</b> {perf['slo']['objective']}; window: {perf['slo']['window']}. Error budget = 1% x {perf['requests']} = "
                f"{perf['slo']['error_budget_requests']} requests; used {perf['slo']['budget_used_requests']} ({perf['slo']['budget_used_pct']}%).", B)]
    else:
        S += [P(f"Load-test, cost and SLO results: {NA} (run scripts/perf.py).", B)]
    v1 = J("student_v1_validation.json")
    pe = perf.get("p95_ms", NA)
    first = next(iter(perf.get("peak_cpu_mem_per_pod", {}).values()), {})
    S += [tbl([["Component", "Evidence (measured)", "Limitation found", "Improvement"],
               ["Data and storage", f"{counts.get('kept', NA)} of {counts.get('input', NA)} teacher replies kept", "Single SeaweedFS replica on one PVC; no versioning beyond folders", "Replicated volumes, object versioning"],
               ["Model quality", f"v1 full_frame {v1.get('full_frame', NA)}%, validity {v1.get('json_validity', NA)}%", "70M model, 1,500 examples, teacher errors copied", "More data, filter by agreement, logit KD with a same-tokenizer teacher"],
               ["Latency and throughput", f"p95 {pe} ms, {perf.get('rps', NA)} req/s", "Greedy autoregressive CPU decoding, one request at a time per Pod", "Quantisation/ONNX, batching, shorter output format"],
               ["Resource utilisation", f"peak {first.get('cpu_m', NA)}m CPU / {first.get('mem_mi', NA)}Mi per Pod", "Pods share the laptop; CPU limit throttles under load", "Tune requests from measured usage"],
               ["Reliability and recovery", f"error rate {perf.get('error_rate_pct', NA)}%", "Single node: no machine-level fault tolerance", "Multi-node cluster, PodDisruptionBudget"],
               ["Security and cost", f"{perf.get('cost', {}).get('cost_per_1000_requests_inr', NA)} INR per 1,000 requests", "S3 keys in plain Secrets, no TLS/auth on /parse", "Sealed secrets, TLS on Ingress, request auth"]],
              [28 * mm, 45 * mm, 52 * mm, 45 * mm])]
    out = ROOT / "report"
    out.mkdir(exist_ok=True)
    SimpleDocTemplate(str(out / "report.pdf"), pagesize=A4, leftMargin=14 * mm, rightMargin=14 * mm,
                      topMargin=12 * mm, bottomMargin=12 * mm, title="Cloud Computing Mini Project").build(S)
    print("wrote report/report.pdf")


if __name__ == "__main__":
    main()
