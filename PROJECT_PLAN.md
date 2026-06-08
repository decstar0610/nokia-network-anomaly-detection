# PROJECT PLAN — Converting the Current Build into the Nokia PRD

**Goal:** Take the existing working NSL-KDD pipeline and evolve it into the system described in `Nokia_Network_Anomaly_Detection_PRD.md` — an **unsupervised, time-series, real-time anomaly detection service with alerting and a REST API.**

**Decisions locked in for this plan:**
- Scope: **Full PRD** (all 6 phases).
- Model strategy: **Replace RandomForest** with the PRD's models — Isolation Forest (primary), LOF (baseline), LSTM autoencoder (time-series).
- Keep the current pipeline scaffolding (load → encode → scale → split → evaluate) and refactor it; don't throw it away.

---

## 0. Reality Check — Read This First

Two facts shape the whole plan, so they're called out up front:

1. **NSL-KDD is not a time-series / streaming dataset.** It is a set of independent *connection records*. It has **no `timestamp`, no `src_ip`/`dst_ip`, no ports** — so the PRD's input schema (§7.1.1) and its temporal/seasonal features (`hour_of_day`, rolling windows) **cannot be derived from NSL-KDD directly.** To satisfy the PRD's time-series and real-time requirements we will **generate a synthetic timestamped stream** (PRD explicitly allows synthetic data, §7.1.3) layered on top of, or instead of, NSL-KDD for the streaming/LSTM portions.

2. **Unsupervised ≠ supervised.** The current 99.9% scores come from a supervised RF memorizing labels. The PRD wants models that learn "normal" and flag deviations. So labels are used **only for evaluation**, never for training the primary detector. Expect lower-but-honest scores (PRD targets: Precision >85%, Recall >80%, AUC >0.90).

We will run **two dataset tracks** in parallel:
- **Track A — NSL-KDD (tabular):** Isolation Forest + LOF, trained on normal-only, evaluated against labels. Covers the "intrusion detection" core.
- **Track B — Synthetic time-series stream:** LSTM autoencoder + streaming simulator + API + alerting. Covers the "real-time / temporal / predictive" core.

---

## 1. Target Repository Structure

```
network-anomaly-detection/
├── data/
│   ├── raw/                       # NSL-KDD + generated synthetic stream
│   ├── processed/                 # X_train, X_test, normal-only train sets
│   └── README.md                  # data dictionary
├── notebooks/
│   ├── 01_EDA.ipynb               # NEW — distributions, patterns, outliers
│   ├── 02_Model_Training.ipynb    # NEW — IF / LOF / LSTM training
│   └── 03_Evaluation.ipynb        # NEW — metrics, ROC, PR, comparison
├── src/
│   ├── data_loader.py             # refactor of 01_load_dataset.py
│   ├── preprocessing.py           # refactor of 02/03 (+ feature engineering)
│   ├── feature_engineering.py     # NEW — packet_rate, temporal, rolling
│   ├── synthetic_stream.py        # NEW — timestamped traffic generator
│   ├── models/
│   │   ├── isolation_forest.py    # NEW — primary
│   │   ├── lof.py                 # NEW — baseline
│   │   └── lstm_autoencoder.py    # NEW — time-series
│   ├── evaluation.py              # refactor of 06_evaluate_model.py
│   ├── streaming_simulator.py     # NEW — replays stream, measures latency
│   ├── alerting.py                # NEW — severity, type, recommendations
│   └── api.py                     # NEW — FastAPI /predict endpoint
├── models/                        # saved .pkl / .h5 + scaler + metadata.json
├── results/                       # metrics.json, confusion_matrix, roc, pr
├── tests/
│   └── test_*.py                  # NEW — unit tests, >80% coverage target
├── Dockerfile                     # NEW
├── requirements.txt               # add tensorflow, fastapi, uvicorn, shap, pytest
├── README.md
└── PROJECT_PLAN.md
```

---

## 2. Phase-by-Phase Plan

Each phase lists: current state → what to build → deliverable → acceptance criteria (tied to PRD requirement IDs).

### Phase 1 — Data Ingestion, Feature Engineering & EDA (Week 1–2)

**Current state:** Loads NSL-KDD, label-encodes 3 categorical cols, StandardScaler, 80/20 stratified split. No feature engineering, no EDA, no notebooks.

**Build:**
- [ ] Refactor `01/02/03/04` scripts into importable modules (`data_loader.py`, `preprocessing.py`).
- [ ] `feature_engineering.py` — derive PRD §7.3 features where possible from NSL-KDD, and the full set on the **synthetic stream**.
- [ ] `synthetic_stream.py` — generate 100k+ timestamped records over 30 simulated days, 95% normal / 5% anomalous, matching PRD §7.1.1 schema.
- [ ] `01_EDA.ipynb` — traffic distributions, class imbalance, temporal patterns, normal-vs-anomalous visualizations.
- [ ] Create a **normal-only training set** (drop anomalies from train) for unsupervised models.

**Acceptance (PRD FR1.1, FR1.2, Phase 2):** 10k+ records loaded & validated; engineered features computed; missing values handled; EDA visualizations produced.

### Phase 2 — Model Development: IF + LOF + LSTM (Week 3–4)

**Build:**
- [ ] `isolation_forest.py` — **primary model**, hyperparameters per PRD §7.2.1. Train on normal-only, score new records, output anomaly confidence 0–1.
- [ ] `lof.py` — **baseline**, PRD §7.2.3.
- [ ] `lstm_autoencoder.py` — **time-series**, PRD §7.2.2 architecture. Runs on the synthetic stream.
- [ ] Hyperparameter tuning + a single `train.py` orchestrator.
- [ ] `02_Model_Training.ipynb`.

**Acceptance (PRD FR2.1, FR2.2, Phase 3):** IF and LOF train without labels; LSTM detects temporal anomalies; each outputs an anomaly score.

### Phase 3 — Evaluation, Validation & Comparison (Week 5)

**Build:**
- [ ] Refactor `evaluation.py` to evaluate **all models on the same held-out test set** using labels.
- [ ] Add **Precision-Recall curve** (critical for 95/5 imbalance), not just ROC.
- [ ] **Model comparison table** (IF vs LOF vs LSTM vs supervised baseline).
- [ ] **Threshold tuning** to hit PRD's FPR <15% (PRD R3).
- [ ] **Feature importance / SHAP** for interpretability (PRD R7).
- [ ] K-fold cross-validation + error analysis.
- [ ] `03_Evaluation.ipynb`.

**Acceptance (PRD §7.4, NFR2):** Report Precision/Recall/F1/AUC per model; FPR measured; targets evaluated honestly.

### Phase 4 — Real-Time Streaming Simulation (Week 6)

**Build:**
- [ ] `streaming_simulator.py` — replays the synthetic stream record-by-record, maintains a **rolling baseline** of normal traffic.
- [ ] **Latency instrumentation** — per-record detection time (PRD NFR1: <100ms).
- [ ] **Throughput stress test** toward PRD NFR1.2.

**Acceptance (PRD FR3.1, Phase 5, NFR1):** Streaming detection works; latency measured per record; rolling baseline maintained.

### Phase 5 — Alerting & REST API (Week 7)

**Build:**
- [ ] `alerting.py` — map anomaly score → **severity (LOW/MEDIUM/HIGH/CRITICAL)**, infer **anomaly type**, attach **confidence** and **recommended action**, emit **JSON** (PRD FR4.1).
- [ ] DDoS detection (spike >3× baseline + source uniqueness, US1), gradual packet-loss trend thresholds (US2).
- [ ] `api.py` — **FastAPI** with `POST /predict` returning anomaly score + alert JSON. Add `/health`.

**Acceptance (PRD FR3.1, FR4.1, US1, US2):** `/predict` returns structured alert with type, severity, confidence, recommendation in JSON.

### Phase 6 — Testing, Packaging & Documentation (Week 8)

**Build:**
- [ ] `tests/` — unit tests for preprocessing, each model, alerting, API (PRD NFR4: >80% coverage).
- [ ] `Dockerfile` + `docker-compose` for local run.
- [ ] Update `README.md`; FastAPI auto-docs.
- [ ] Update `requirements.txt`.
- [ ] Refresh `report/Report.docx` with final metrics and Nokia-alignment points (PRD §12).

**Acceptance (PRD §11.3, NFR4):** Tests pass with >80% coverage; `docker build` works; docs complete.

---

## 3. Gap → Phase Traceability

| PRD requirement | Status now | Closed in |
|---|---|---|
| Isolation Forest (primary) | ❌ Missing | Phase 2 |
| LOF (baseline) | ❌ Missing | Phase 2 |
| LSTM time-series (FR2.2) | ❌ Missing | Phase 2 |
| Unsupervised (train on normal) | ❌ Supervised RF | Phase 2 |
| Feature engineering (§7.3) | ❌ Raw 41 cols | Phase 1 |
| EDA notebooks | ❌ Scripts only | Phase 1/3 |
| Real-time streaming (FR3.1) | ❌ Missing | Phase 4 |
| Latency / throughput (NFR1) | ❌ Not measured | Phase 4 |
| Alerting w/ severity & type (FR4.1) | ❌ Missing | Phase 5 |
| REST API (Flask/FastAPI) | ❌ CLI only | Phase 5 |
| Model comparison + PR curve | ❌ One model | Phase 3 |
| SHAP / interpretability (R7) | ❌ Missing | Phase 3 |
| Unit tests >80% (NFR4) | ❌ None | Phase 6 |
| Docker | ❌ None | Phase 6 |
| ✅ Data load/scale/split | Working | Refactored, kept |
| ✅ Core metrics + ROC | Working | Extended in Phase 3 |
| ✅ Clean repo + README | Working | Updated Phase 6 |

---

## 4. Suggested Order of Execution (Critical Path)

1. **Synthetic stream generator first** (Phase 1) — it unblocks LSTM, streaming, API, and alerting. Do not defer it.
2. Feature engineering + EDA.
3. Isolation Forest → LOF (fast wins, tabular track).
4. LSTM autoencoder (time-series track).
5. Evaluation + threshold tuning + comparison.
6. Streaming simulator → API → alerting (these chain together).
7. Tests + Docker + docs last.

## 5. Key Risks

- **Unsupervised scores drop sharply** vs the current 99.9%. Expected and correct — frame it honestly. Mitigate with threshold tuning (R3).
- **NSL-KDD lacks temporal fields** → reliance on synthetic stream for Track B. Document this assumption clearly.
- **LSTM scope creep**. MVP fallback is IF + LOF + API + alerting; LSTM is the stretch goal (PRD §11.2).
- **Throughput target (10k rec/s)** is ambitious on CPU — measure and report real numbers.

## 6. Definition of Done (PRD MVP §11.1 + full)

- [ ] Isolation Forest with tuned threshold, FPR <15%, AUC reported.
- [ ] LOF baseline + LSTM time-series + comparison table.
- [ ] Engineered features + EDA notebooks.
- [ ] Streaming simulator with measured latency.
- [ ] FastAPI `/predict` returning JSON alerts with severity/type/recommendation.
- [ ] Tests >80% coverage, Dockerfile, full README + API docs.
- [ ] Report.docx updated with honest metrics and Nokia-alignment narrative.
