# Fraud GenAI Explainer (MVP)

This project trains a baseline fraud scorer (LightGBM) on the public credit card dataset,
enriches it with synthetic entity fields (merchant/device/IP/geo), computes SHAP drivers,
builds **Evidence JSON**, and generates two grounded explanations (analyst + plain language).

The goal is to demonstrate how traditional ML scoring + explainability can be combined
with structured evidence and GenAI-style narrative outputs for fraud investigations.

---

## Setup

### 1) Download the Dataset

Download the public credit card fraud dataset and place it here:

```
data/creditcard.csv
```

> The dataset is not included in this repository due to size and licensing considerations.

---

### 2) Create Virtual Environment + Install Dependencies

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you're on macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

---

### 3) Run the Project

```powershell
python .\run.py
```

---

## Outputs

Running the pipeline will generate:

- `outputs/alerts_with_evidence.jsonl`
- `outputs/llm_reports.jsonl`

These contain:
- Model scores
- Structured evidence objects
- Generated explanation reports

---

## Notes / Grey Areas

- The LLM call is currently stubbed in `src/llm.py` (deterministic template output).
- Replace the stub with a real API call if you want live GenAI-generated explanations.
- Synthetic entity fields (merchant/device/IP/geo) are simulated for demo/demo-quality purposes.
- This is an MVP-style explainer project, not a production-grade fraud detection system.
- Performance tuning, threshold calibration, and monitoring are intentionally minimal.
