import json
import joblib
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from .config import Config
from .enrich import enrich_transactions
from .features import add_derived_features, compute_train_only_rates, apply_rates
from .explain import compute_shap_for_alerts
from .evidence import build_evidence_row, write_jsonl
from .llm import generate_explanations_with_llm, validate_json
from .train import train_pipeline

def run_all(csv_path: str):
    cfg = Config()

    model_path = "artifacts/model.pkl"
    enc_path = "artifacts/encoders.pkl"

    # Train model
    train_pipeline(csv_path, cfg, model_path, enc_path)

    # Rebuild dataset (same enrichment + features) and create alerts on held-out split
    df = pd.read_csv(csv_path)
    df, _ = enrich_transactions(df, cfg)
    df = add_derived_features(df)

    train_df, test_df = train_test_split(
        df,
        test_size=cfg.test_size,
        random_state=cfg.seed,
        stratify=df["Class"]
    )

    rates = compute_train_only_rates(train_df)
    joblib.dump(rates, "artifacts/rates.pkl")   # <-- ADD THIS LINE
    train_df = apply_rates(train_df, rates)
    test_df = apply_rates(test_df, rates)

    model = joblib.load(model_path)
    enc = joblib.load(enc_path)
    features = enc["features"]

    X_test = test_df[features]
    p = model.predict_proba(X_test)[:, 1]
    test_df = test_df.copy()
    test_df["risk_score"] = p

    # Alert selection: top K above threshold; if none, fallback to top K scores
    alerts = test_df[test_df["risk_score"] >= cfg.alert_threshold].sort_values("risk_score", ascending=False)
    alerts = alerts.head(cfg.top_k_alerts)
    if len(alerts) == 0:
        alerts = test_df.sort_values("risk_score", ascending=False).head(cfg.top_k_alerts)

    # SHAP for selected alerts
    shap_values, feat_names = compute_shap_for_alerts(model_path, enc_path, alerts)

    # Evidence JSONL
    evidences = []
    for i, (idx, row) in enumerate(alerts.iterrows()):
        ev = build_evidence_row(row, row["risk_score"], shap_values[i], feat_names)
        evidences.append(ev)

    write_jsonl(evidences, "outputs/alerts_with_evidence.jsonl")

    # Generate reports JSONL (LLM stub)
    reports = []
    for ev in tqdm(evidences, desc="Generating explanations"):
        rep = generate_explanations_with_llm(ev)
        validate_json(rep)
        reports.append({"alert_id": ev["alert_id"], "report": rep})

    with open("outputs/llm_reports.jsonl", "w", encoding="utf-8") as f:
        for r in reports:
            f.write(json.dumps(r) + "\n")

    print("Wrote outputs/alerts_with_evidence.jsonl and outputs/llm_reports.jsonl")


