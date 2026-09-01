# evaluate_on_benchmark.py
#
# DEFINITIVE FINAL VERSION
#
# Supports both v1 and v2 models:
#   v1 AraELECTRA: loads best_model_state_dict.pt with electra->transformer remapping
#   v2 models (AraELECTRA + AraBERT): loads from best checkpoint/model.safetensors
#     (keys already use transformer.* — no remapping needed)
#
# Features keyed by preprocessed text — guaranteed alignment.

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


def find_best_checkpoint(model_path):
    """Find best checkpoint folder from trainer_state.json."""
    for ckpt in sorted(Path(model_path).glob("checkpoint-*")):
        state_file = ckpt / "trainer_state.json"
        if state_file.exists():
            state = json.load(open(str(state_file)))
            best = state.get("best_model_checkpoint", None)
            if best:
                best_path = Path(model_path) / Path(best).name
                if best_path.exists():
                    return best_path
            return ckpt  # fallback to first checkpoint found
    return None


def load_model_correctly(model_path, device):
    """
    Loads AdvancedHybridModel correctly for both v1 and v2 models.

    v1 AraELECTRA: has best_model_state_dict.pt — load with electra->transformer remap
    v2 models: has model.safetensors in checkpoint — keys already transformer.*
    """
    model_path = Path(model_path)
    best_pt = model_path / "best_model_state_dict.pt"

    if best_pt.exists():
        # ── v1 AraELECTRA loading ─────────────────────────────
        print("Detected v1 AraELECTRA model (best_model_state_dict.pt found)")
        print(f"Loading config from: {model_path}")
        config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
        config._attn_implementation = "eager"
        loaded_model = model.AdvancedHybridModel(config)

        print(f"Loading weights from: {best_pt}")
        raw_state_dict = torch.load(str(best_pt), map_location=device)

        new_state_dict = {}
        remapped = 0
        for key, value in raw_state_dict.items():
            if key.startswith("electra."):
                new_key = key.replace("electra.", "transformer.", 1)
                remapped += 1
            elif key.startswith("bert."):
                new_key = key.replace("bert.", "transformer.", 1)
                remapped += 1
            else:
                new_key = key
            new_state_dict[new_key] = value

        if remapped > 0:
            print(f"Remapped {remapped} keys to transformer.* namespace.")

        missing, unexpected = loaded_model.load_state_dict(
            new_state_dict, strict=False
        )
        print(f"Weights loaded: Missing={len(missing)}, Unexpected={len(unexpected)}")
        if len(missing) == 0:
            print("All weights loaded perfectly!")

    else:
        # ── v2 model loading (AraELECTRA v2 or AraBERT) ──────
        print("Detected v2 model (no best_model_state_dict.pt)")

        # Find best checkpoint
        best_ckpt = find_best_checkpoint(model_path)

        if best_ckpt and (best_ckpt / "model.safetensors").exists():
            load_path = best_ckpt
            print(f"Loading from best checkpoint: {best_ckpt.name}")
        elif (model_path / "model.safetensors").exists():
            load_path = model_path
            print(f"Loading from final model folder: {model_path.name}")
        else:
            print("ERROR: No weights file found.")
            return None

        print(f"Loading config from: {load_path}")
        config = AutoConfig.from_pretrained(str(load_path), local_files_only=True)
        config._attn_implementation = "eager"
        loaded_model = model.AdvancedHybridModel(config)

        weights_file = load_path / "model.safetensors"
        print(f"Loading weights from: {weights_file}")
        weights = load_safetensors(str(weights_file))

        # Check key prefixes
        prefixes = set(k.split(".")[0] for k in weights.keys())
        print(f"Key prefixes in weights: {prefixes}")

        # Remap if needed (should not be needed for v2 but just in case)
        if "electra" in prefixes or "bert" in prefixes:
            new_weights = {}
            remapped = 0
            for k, v in weights.items():
                if k.startswith("electra."):
                    new_weights[k.replace("electra.", "transformer.", 1)] = v
                    remapped += 1
                elif k.startswith("bert."):
                    new_weights[k.replace("bert.", "transformer.", 1)] = v
                    remapped += 1
                else:
                    new_weights[k] = v
            if remapped > 0:
                print(f"Remapped {remapped} keys to transformer.* namespace.")
            weights = new_weights

        missing, unexpected = loaded_model.load_state_dict(weights, strict=False)
        print(f"Weights loaded: Missing={len(missing)}, Unexpected={len(unexpected)}")
        if len(missing) == 0:
            print("All weights loaded perfectly!")
        else:
            print(f"WARNING: {len(missing)} missing keys. First: {missing[0]}")

    # Verify classifier bias
    bias = loaded_model.classifier[4].bias.data
    print(f"Classifier bias: [{bias[0]:.6f}, {bias[1]:.6f}]")
    if bias[1] > bias[0]:
        print("Bias check PASSED — model favors AI for class 1.")
    else:
        print("WARNING: Unexpected bias direction — predictions may be inverted.")

    loaded_model.to(device)
    loaded_model.eval()
    return loaded_model


def main():
    print("=" * 60)
    print("  AIRABIC Benchmark Evaluation — DEFINITIVE VERSION")
    print(f"  Model    : {project_config.PRE_TRAINED_MODEL_NAME}")
    print(f"  Features : {project_config.NUM_EXPLICIT_FEATURES}")
    print(f"  Dir      : {project_config.SAVED_MODEL_DIR}")
    print("=" * 60)

    # ── 1. Paths ─────────────────────────────────────────────
    model_path  = Path(project_config.SAVED_MODEL_DIR).resolve()
    scaler_path = Path(project_config.SCALER_PATH).resolve()
    device      = project_config.DEVICE

    if not (model_path / "config.json").exists():
        print(f"ERROR: Cannot find config.json in {model_path}")
        return

    # ── 2. Load model ─────────────────────────────────────────
    loaded_model = load_model_correctly(model_path, device)
    if loaded_model is None:
        return

    # ── 3. Load tokenizer and scaler ─────────────────────────
    tokenizer = processing.get_tokenizer(str(model_path))
    scaler    = joblib.load(str(scaler_path))
    print(f"\nScaler loaded. Expects {scaler.n_features_in_} features.")

    if scaler.n_features_in_ != project_config.NUM_EXPLICIT_FEATURES:
        print(f"ERROR: Scaler/config mismatch: "
              f"{scaler.n_features_in_} vs {project_config.NUM_EXPLICIT_FEATURES}")
        return

    # ── 4. Load benchmark CSV ────────────────────────────────
    # Use airabic_holdout_test.csv for v2 models (fair evaluation)
    # Use airabic.csv for original benchmark
    holdout_path = project_config.DATA_DIR / "airabic_holdout_test_v3.csv"
    original_path = project_config.DATA_DIR / "airabic.csv"

    if holdout_path.exists():
        benchmark_path = holdout_path
        print(f"\nUsing HELD-OUT test set: {benchmark_path}")
    else:
        benchmark_path = original_path
        print(f"\nUsing full AIRABIC: {benchmark_path}")

    df = pd.read_csv(str(benchmark_path))
    print("Label distribution:", df["type"].value_counts().to_dict())
    print("(Expected: 0=Human, 1=AI)")

    # ── 5. Preprocess text ────────────────────────────────────
    print("\nPreprocessing text...")
    df["text"] = df["artical_text"].astype(str).fillna("").progress_apply(
        processing.light_preprocess_for_llm
    )
    df = df[df["text"].str.strip().astype(bool)].reset_index(drop=True)
    df["true_label"] = df["type"].astype(int)
    print(f"Total samples: {len(df)}")

    # ── 6. Load or calculate features ────────────────────────
    parquet_path = project_config.DATA_DIR / "airabic_with_features.parquet"
    feat_cols    = project_config.EXPLICIT_FEATURE_COLUMNS
    features_array = None

    if parquet_path.exists():
        print(f"\nLoading features from: {parquet_path}")
        try:
            feat_df_saved = pd.read_parquet(str(parquet_path))
            missing_cols  = [c for c in feat_cols
                             if c not in feat_df_saved.columns]
            has_text_key  = "text" in feat_df_saved.columns

            if not missing_cols and has_text_key:
                n_before  = len(df)
                df_merged = df.merge(
                    feat_df_saved[["text"] + feat_cols],
                    on="text", how="left"
                )
                n_missing = df_merged[feat_cols[0]].isna().sum()
                if n_missing == 0 and len(df_merged) == n_before:
                    df = df_merged
                    features_array = df[feat_cols].values.astype(float)
                    print(f"Features loaded and aligned. Shape: {features_array.shape}")
                else:
                    print(f"Alignment issue — recalculating features...")
            else:
                if missing_cols:
                    print(f"Missing columns in parquet: {missing_cols}")
                if not has_text_key:
                    print("Parquet missing text key.")
                print("Will recalculate features...")
        except Exception as e:
            print(f"Could not load parquet: {e}. Recalculating...")

    if features_array is None:
        print("\nCalculating features (~30-60 min)...")
        feature_engineering.initialize_feature_engineering()
        feature_dicts  = df["text"].progress_apply(
            feature_engineering.calculate_features
        ).tolist()
        feat_df_calc   = pd.DataFrame(feature_dicts)
        features_array = feat_df_calc[feat_cols].values.astype(float)
        save_df = df[["text", "true_label"]].copy()
        for col in feat_cols:
            save_df[col] = feat_df_calc[col].values
        save_df.to_parquet(str(parquet_path), index=False)
        print(f"Features saved to {parquet_path}.")

    features_scaled = scaler.transform(
        pd.DataFrame(features_array, columns=feat_cols)
    )
    print(f"Features scaled. Shape: {features_scaled.shape}")

    # ── 7. Run inference ──────────────────────────────────────
    print("\nRunning inference...")
    texts      = df["text"].tolist()
    y_true     = df["true_label"].values
    batch_size = 16
    n          = len(texts)
    all_preds  = []
    all_scores = []

    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc="Evaluating"):
            end = min(start + batch_size, n)

            inputs = tokenizer(
                texts[start:end],
                padding="max_length",
                truncation=True,
                max_length=project_config.MAX_LENGTH,
                return_tensors="pt",
            ).to(device)

            feat_tensor = torch.tensor(
                features_scaled[start:end], dtype=torch.float
            ).to(device)

            outputs = loaded_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs.get("token_type_ids"),
                explicit_features=feat_tensor,
            )

            logits = outputs.logits.cpu().numpy()
            probs  = softmax(logits, axis=1)
            preds  = np.argmax(logits, axis=1)
            scores = probs[:, 1]

            all_preds.extend(preds.tolist())
            all_scores.extend(scores.tolist())

    y_pred  = np.array(all_preds)
    y_score = np.array(all_scores)



    # ── 9. Compute metrics ────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy    = accuracy_score(y_true, y_pred)
    bal_acc     = balanced_accuracy_score(y_true, y_pred)
    precision   = precision_score(y_true, y_pred, zero_division=0)
    recall      = recall_score(y_true, y_pred, zero_division=0)
    f1          = f1_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    npv         = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
    fpr_val     = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    fnr_val     = fn / (fn + tp) if (fn + tp) > 0 else float("nan")
    mcc         = matthews_corrcoef(y_true, y_pred)
    kappa       = cohen_kappa_score(y_true, y_pred)
    roc_auc     = roc_auc_score(y_true, y_score)
    pr_auc      = average_precision_score(y_true, y_score)

    # ── 10. Build model name ──────────────────────────────────
    base_model = (
        "AraBERT" if "arabertv2" in project_config.PRE_TRAINED_MODEL_NAME
        else "AraELECTRA"
    )
    model_name = f"{base_model}_{project_config.NUM_EXPLICIT_FEATURES}feat"

    # ── 11. Print results ─────────────────────────────────────
    SEP = "=" * 60
    print(f"\n{SEP}")
    print(f"  RESULTS: {model_name} on AIRABIC Benchmark")
    print(f"  Convention: 0=Human, 1=AI")
    print(SEP)
    print(f"  Accuracy               : {accuracy:.4f}")
    print(f"  Balanced Accuracy      : {bal_acc:.4f}")
    print(f"  Precision              : {precision:.4f}")
    print(f"  Recall (Sensitivity)   : {recall:.4f}")
    print(f"  F1-Score               : {f1:.4f}")
    print(f"  Specificity (TNR)      : {specificity:.4f}")
    print(f"  NPV                    : {npv:.4f}")
    print(f"  FPR                    : {fpr_val:.4f}")
    print(f"  FNR                    : {fnr_val:.4f}")
    print(f"  MCC                    : {mcc:.4f}")
    print(f"  Cohen Kappa            : {kappa:.4f}")
    print(f"  ROC-AUC                : {roc_auc:.4f}")
    print(f"  PR-AUC                 : {pr_auc:.4f}")
    print(SEP)
    print(f"  Confusion Matrix")
    print(f"               Human    AI")
    print(f"  Human (0)    {tn:5d}   {fp:5d}   TN={tn}  FP={fp}")
    print(f"  AI    (1)    {fn:5d}   {tp:5d}   FN={fn}  TP={tp}")
    print(SEP)
    print("\nDetailed Classification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=["Human (0)", "AI (1)"],
        digits=4, zero_division=0
    ))

    # ── 12. Save results ──────────────────────────────────────
    os.makedirs("benchmark_results", exist_ok=True)

    metrics = {
        "Model":             model_name,
        "Accuracy":          round(accuracy,    4),
        "Balanced Accuracy": round(bal_acc,     4),
        "Precision":         round(precision,   4),
        "Recall":            round(recall,      4),
        "F1-Score":          round(f1,          4),
        "Specificity":       round(specificity, 4),
        "NPV":               round(npv,         4),
        "FPR":               round(fpr_val,     4),
        "FNR":               round(fnr_val,     4),
        "MCC":               round(mcc,         4),
        "Cohen Kappa":       round(kappa,       4),
        "ROC-AUC":           round(roc_auc,     4),
        "PR-AUC":            round(pr_auc,      4),
        "TP": int(tp), "FP": int(fp),
        "FN": int(fn), "TN": int(tn),
    }

    json_path = f"benchmark_results/{model_name}_airabic_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to: {json_path}")

    pred_df = pd.DataFrame({
        "true_label": y_true,
        "pred_label": y_pred,
        "ai_score":   y_score,
    })
    csv_path = f"benchmark_results/{model_name}_airabic_predictions.csv"
    pred_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Predictions saved to: {csv_path}")

    comparison_path = "benchmark_results/all_models_comparison.csv"
    if os.path.exists(comparison_path):
        comp_df = pd.read_csv(comparison_path)
        comp_df = comp_df[comp_df["Model"] != model_name]
        comp_df = pd.concat(
            [comp_df, pd.DataFrame([metrics])], ignore_index=True
        )
    else:
        comp_df = pd.DataFrame([metrics])
    comp_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    print(f"\n{SEP}")
    print("  COMPARISON TABLE (all experiments so far)")
    print(SEP)
    display = ["Model", "Accuracy", "Precision", "Recall",
               "F1-Score", "Specificity", "MCC", "ROC-AUC"]
    print(comp_df[[c for c in display if c in comp_df.columns]].to_string(
        index=False
    ))
    print(SEP)
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
