# NLP Ticket Triage & Sentiment

Compact transformer to auto‑label help‑desk tickets (**topic + sentiment**) with a **FastAPI** endpoint and a **Streamlit** eval dashboard. Ships with PII redaction, tests, Docker, and CI.

---

## ✨ Features
- **Multi‑task head**: topic (multi‑class) + sentiment (neg/neu/pos)
- **FastAPI** `/predict`, `/feedback`, `/health`
- **Eval dashboard** (Streamlit): AUC‑PR, F1, confusion, error explorer
- **MLOps glue**: MLflow tracking, CI (lint, test, build), Docker image
- **PII safety**: email/phone/ticket ID redaction before logging

## 🚀 Quickstart
```bash
# 0) Python 3.10+
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Run API
uvicorn app.main:app --host 0.0.0.0 --port 8000
# -> http://localhost:8000/docs

# 2) Run tests
pytest -q

# 3) (Optional) Dashboard
streamlit run src/eval/dashboard.py  # stub provided; fill later
```

## 📡 API
**POST /predict**
```json
{ "text": "Refund failed twice; card charged." }
```
**Response**
```json
{
  "topic": {"label":"billing","score":0.82},
  "sentiment": {"label":"neg","score":0.91},
  "probs": {"topic": {"billing":0.82}, "sentiment": {"neg":0.91}},
  "latency_ms": 14,
  "model_version": "distilbert-l6-2025-11-03"
}
```

## 🗂️ Repo Structure
```
nlp-ticket-triage/
├─ app/
│  └─ main.py
├─ src/
│  ├─ infer/
│  │  ├─ preprocess.py
│  │  └─ service.py
│  ├─ models/
│  │  └─ multitask_head.py
│  └─ eval/
│     └─ dashboard.py (stub)
├─ tests/
│  └─ test_service.py
├─ scripts/
│  └─ serve.sh
├─ .github/workflows/ci.yml
├─ requirements.txt
├─ Dockerfile
├─ .env.example
└─ .gitignore
```

## 🔧 Training (placeholder)
A simple `MultiTask` PyTorch module is provided; replace the mock infer logic with your trained head when ready. Suggested backbone: `distilbert-base-uncased` or `microsoft/MiniLM-L6-v2`.

## 🔐 PII Redaction
Before any logging or feedback persistence, `preprocess.py` removes emails, phones, and IDs. Extend as needed.

## 🧪 CI
GitHub Actions runs lint + tests on pushes and PRs. Adjust Python matrix as needed.

## 🐳 Docker
```bash
docker build -t triage:latest .
docker run -p 8000:8000 triage:latest
```

## 🏷️ Topics (GitHub)
`nlp` · `text-classification` · `transformers` · `helpdesk` · `ticket-triage` · `sentiment-analysis` · `fastapi` · `streamlit` · `mlops` · `distilbert` · `minilm` · `huggingface`

---

## 📄 License
MIT (change if you prefer).


---

## 🧪 Train a tiny checkpoint (demo)
```bash
# from repo root
python -m src.models.train --train_csv data/seed/seed.csv --epochs 1 --batch_size 16 --lr 2e-5
# This saves artifacts to artifacts/model/
# Restart the API and /predict will use the real model automatically.
```
