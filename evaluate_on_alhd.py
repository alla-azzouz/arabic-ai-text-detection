# evaluate_on_alhd.py
#
# Evaluates trained model on ALHD benchmark dataset


import pandas as pd
import joblib
import numpy as np
import os
import json
import torch
from pathlib import Path
from tqdm import tqdm
from scipy.special import softmax
from transformers import AutoConfig
from safetensors.torch import load_file as load_safetensors
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, matthews_corrcoef,
    balanced_accuracy_score, cohen_kappa_score,
)

from src import config as project_config
from src import processing
from src import model
from src import feature_engineering

tqdm.pandas()


def load_model_correctly(model_path, device):
    model_path = Path(model_path)
    best_pt = model_path / "best_model_state_dict.pt"

    if best_pt.exists():
        print(f"Loading from best_model_state_dict.pt...")
        config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
        config._attn_implementation = "eager"
        loaded_model = model.AdvancedHybridModel(config)
        raw = torch.load(str(best_pt), map_location=device)
        new = {}
        for k, v in raw.items():
            if k.startswith("electra."): new[k.replace("electra.", "transformer.", 1)] = v
            elif k.startswith("bert."): new[k.replace("bert.", "transformer.", 1)] = v
            else: new[k] = v
        missing, unexpected = loaded_model.load_state_dict(new, strict=False)
        print(f"Missing={len(missing)}, Unexpected={len(unexpected)}")
    else:
        print("Loading from model.safetensors...")
        config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
        config._attn_implementation = "eager"
        loaded_model = model.AdvancedHybridModel(config)
        weights = load_safetensors(str(model_path / "model.safetensors"))
        missing, unexpected = loaded_model.load_state_dict(weights, strict=False)
        print(f"Missing={len(missing)}, Unexpected={len(unexpected)}")

    loaded_model.to(device)
    loaded_model.eval()
    return loaded_model


def main():
    print("=" * 60)
    print("  ALHD Benchmark Evaluation")
    print(f"  Model    : {project_config.PRE_TRAINED_MODEL_NAME}")
    print(f"  Features : {project_config.NUM_EXPLICIT_FEATURES}")
    print(f"  Dir      : {project_config.SAVED_MODEL_DIR}")
    print("=" * 60)

    model_path  = Path(project_config.SAVED_MODEL_DIR).resolve()
    scaler_path = Path(project_config.SCALER_PATH).resolve()
    device      = project_config.DEVICE

    # ── Load model ────────────────────────────────────────────
    loaded_model = load_model_correctly(model_path, device)

    # ── Load tokenizer and scaler ─────────────────────────────
    tokenizer = processing.get_tokenizer(str(model_path))
    scaler    = joblib.load(str(scaler_path))
    print(f"Scaler expects {scaler.n_features_in_} features.")

    # ── Load ALHD benchmark ───────────────────────────────────
    alhd_path = project_config.DATA_DIR / "alhd_holdout_test.csv"
    if not alhd_path.exists():
        print(f"ERROR: ALHD file not found at {alhd_path}")
        return

    df = pd.read_csv(str(alhd_path))
    print(f"\nALHD loaded: {df.shape}")

    # Normalize columns — ALHD uses 'text' and 'label'
    if 'text' in df.columns:
        df['artical_text'] = df['text']
    if 'label' in df.columns:
        df['type'] = df['label']

    print(f"Label distribution: {df['type'].value_counts().to_dict()}")
    print("(0=Human, 1=AI)")

    # Show category breakdown
    if 'subcategory' in df.columns:
        print(f"\nSubcategory breakdown:")
        print(df['subcategory'].value_counts().head(10))

    # ── Preprocess ────────────────────────────────────────────
    print("\nPreprocessing text...")
    df["text_processed"] = df["artical_text"].astype(str).fillna("").progress_apply(
        processing.light_preprocess_for_llm)
    df = df[df["text_processed"].str.strip().astype(bool)].reset_index(drop=True)
    df["true_label"] = df["type"].astype(int)
    print(f"Total samples after cleaning: {len(df)}")

    # ── Calculate or load features ────────────────────────────
    feat_cols    = project_config.EXPLICIT_FEATURE_COLUMNS
    parquet_path = project_config.DATA_DIR / "alhd_with_features.parquet"
    features_array = None

    if parquet_path.exists():
        print(f"Loading cached ALHD features...")
        try:
            cached = pd.read_parquet(str(parquet_path))
            missing_cols = [c for c in feat_cols if c not in cached.columns]
            if not missing_cols and "text_processed" in cached.columns:
                merged = df.merge(cached[["text_processed"] + feat_cols],
                                  on="text_processed", how="left")
                if merged[feat_cols[0]].isna().sum() == 0:
                    df = merged
                    features_array = df[feat_cols].values.astype(float)
                    print(f"Features loaded from cache: {features_array.shape}")
        except Exception as e:
            print(f"Cache load failed: {e}")

    if features_array is None:
        print("Calculating features...")
        feature_engineering.initialize_feature_engineering()
        feat_dicts = df["text_processed"].progress_apply(
            feature_engineering.calculate_features).tolist()
        feat_df = pd.DataFrame(feat_dicts)
        features_array = feat_df[feat_cols].values.astype(float)

        # Save cache
        save_df = df[["text_processed", "true_label"]].copy()
        for col in feat_cols:
            save_df[col] = feat_df[col].values
        save_df.to_parquet(str(parquet_path), index=False)
        print(f"Features cached to {parquet_path}")

    features_scaled = scaler.transform(pd.DataFrame(features_array, columns=feat_cols))

    # ── Inference ─────────────────────────────────────────────
    print("\nRunning inference...")
    texts      = df["text_processed"].tolist()
    y_true     = df["true_label"].values
    batch_size = 16
    all_preds  = []
    all_scores = []

    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="Evaluating"):
            end = min(start + batch_size, len(texts))
            inputs = tokenizer(
                texts[start:end], padding="max_length", truncation=True,
                max_length=project_config.MAX_LENGTH, return_tensors="pt").to(device)
            feat_tensor = torch.tensor(
                features_scaled[start:end], dtype=torch.float).to(device)
            outputs = loaded_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs.get("token_type_ids"),
                explicit_features=feat_tensor)
            logits = outputs.logits.cpu().numpy()
            probs  = softmax(logits, axis=1)
            all_preds.extend(np.argmax(logits, axis=1).tolist())
            all_scores.extend(probs[:, 1].tolist())

    y_pred  = np.array(all_preds)
    y_score = np.array(all_scores)

 

    # ── Metrics ───────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy    = accuracy_score(y_true, y_pred)
    precision   = precision_score(y_true, y_pred, zero_division=0)
    recall      = recall_score(y_true, y_pred, zero_division=0)
    f1          = f1_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    mcc         = matthews_corrcoef(y_true, y_pred)
    roc_auc     = roc_auc_score(y_true, y_score)
    bal_acc     = balanced_accuracy_score(y_true, y_pred)

    base_model = "AraBERT" if "arabertv2" in project_config.PRE_TRAINED_MODEL_NAME else "AraELECTRA"
    model_name = f"{base_model}_{project_config.NUM_EXPLICIT_FEATURES}feat"

    SEP = "=" * 60
    print(f"\n{SEP}")
    print(f"  RESULTS: {model_name} on ALHD Benchmark")
    print(f"  Convention: 0=Human, 1=AI")
    print(SEP)
    print(f"  Accuracy          : {accuracy:.4f}")
    print(f"  Balanced Accuracy : {bal_acc:.4f}")
    print(f"  Precision         : {precision:.4f}")
    print(f"  Recall            : {recall:.4f}")
    print(f"  F1-Score          : {f1:.4f}")
    print(f"  Specificity       : {specificity:.4f}")
    print(f"  MCC               : {mcc:.4f}")
    print(f"  ROC-AUC           : {roc_auc:.4f}")
    print(SEP)
    print(f"  Confusion Matrix")
    print(f"           Human    AI")
    print(f"  Human    {tn:5d}   {fp:5d}   TN={tn}  FP={fp}")
    print(f"  AI       {fn:5d}   {tp:5d}   FN={fn}  TP={tp}")
    print(SEP)
    print(classification_report(y_true, y_pred,
        target_names=["Human (0)", "AI (1)"], digits=4, zero_division=0))

    # ── Save results ──────────────────────────────────────────
    os.makedirs("benchmark_results", exist_ok=True)
    metrics = {
        "Model": model_name, "Benchmark": "ALHD",
        "Accuracy": round(accuracy, 4),
        "Balanced_Accuracy": round(bal_acc, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "Specificity": round(specificity, 4),
        "MCC": round(mcc, 4),
        "ROC_AUC": round(roc_auc, 4),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }
    out_path = f"benchmark_results/{model_name}_ALHD_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {out_path}")
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()