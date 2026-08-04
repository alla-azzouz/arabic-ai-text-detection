# tune_threshold.py

import torch
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader


tqdm.pandas()
# ==========================================================

# ===== Project modules =====
from src import config as project_config
from src import processing
from src import model
from src import feature_engineering
from src.train import set_seed, create_dataset, CustomDataCollatorWithFeatures

def get_predictions_manual(model_obj, dataloader, device):
    """
    Helper function to get model predictions using a manual PyTorch loop.
    """
    print(f"Getting predictions for {len(dataloader.dataset)} samples...")
    model_obj.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            labels = batch.pop("labels").cpu().numpy()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model_obj(**batch)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            ai_probabilities = probabilities[:, project_config.LABEL_MAP["ai"]]
            all_probs.extend(ai_probabilities)
            all_labels.extend(labels)

    return np.array(all_probs), np.array(all_labels)

def main():
    set_seed()
    print("--- Starting Decision Threshold Tuning ---")

    # --- 1. Load Components ---
    print("Loading the best trained model and components...")
    model_path = str(project_config.SAVED_MODEL_DIR)
    loaded_model = model.AdvancedHybridModel.from_pretrained(model_path)
    loaded_model.to(project_config.DEVICE)
    loaded_model.eval()

    tokenizer = processing.get_tokenizer(model_path)
    scaler = joblib.load(project_config.SCALER_PATH)
    collator = CustomDataCollatorWithFeatures(tokenizer=tokenizer)

    # --- 2. Prepare VALIDATION Dataset ---
    print("\n--- Preparing Validation Set ---")
    full_df = pd.read_parquet(project_config.DATA_DIR / "dataset_with_features.parquet")
    _, val_df, _ = processing.split_data(full_df)

    X_val_scaled = scaler.transform(val_df[project_config.EXPLICIT_FEATURE_COLUMNS])
    val_dataset = create_dataset(val_df, X_val_scaled)
    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=project_config.MAX_LENGTH)
    tokenized_val_ds = val_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    val_dataloader = DataLoader(tokenized_val_ds, batch_size=project_config.BATCH_SIZE * 4, collate_fn=collator)
    
   
    val_probs, val_labels = get_predictions_manual(loaded_model, val_dataloader, project_config.DEVICE)

    # --- 3. Find Best Threshold ---
    print("\n--- Searching for the best F1-maximizing threshold... ---")
    best_f1 = 0
    best_threshold = 0.5
    for threshold in tqdm(np.arange(0.30, 0.71, 0.01), desc="Sweeping Thresholds"):
        preds = (val_probs >= threshold).astype(int)
        f1 = f1_score(val_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"\n--- Best Threshold Found ---")
    print(f"  - Optimal Threshold: {best_threshold:.2f}")
    print(f"  - Achieved Validation F1: {best_f1:.4f}")

    # --- 4. Apply to BENCHMARK Set ---
    print("\n--- Applying optimal threshold to the BENCHMARK dataset ---")
    benchmark_df = pd.read_csv(project_config.DATA_DIR / "airabic.csv")
    benchmark_df.rename(columns={"artical_text": "raw_text", "type": "label_id"}, inplace=True)
    if 'model' not in benchmark_df.columns:
        benchmark_df['model'] = 'unknown'
    benchmark_df["raw_text"] = benchmark_df["raw_text"].astype(str).fillna('')
    benchmark_df["text"] = benchmark_df["raw_text"].apply(processing.light_preprocess_for_llm)
    
    print("Calculating features for the benchmark dataset...")
    feature_dicts = benchmark_df['raw_text'].progress_apply(feature_engineering.calculate_features).tolist()
    features_df = pd.DataFrame(feature_dicts)
    required_cols = project_config.EXPLICIT_FEATURE_COLUMNS
    for col in required_cols:
        if col not in features_df.columns: features_df[col] = 0
    X_benchmark_scaled = scaler.transform(features_df[required_cols])
    
    benchmark_dataset = create_dataset(benchmark_df, X_benchmark_scaled)
    tokenized_benchmark_ds = benchmark_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    
    benchmark_dataloader = DataLoader(tokenized_benchmark_ds, batch_size=project_config.BATCH_SIZE * 4, collate_fn=collator)
    benchmark_probs, benchmark_labels = get_predictions_manual(loaded_model, benchmark_dataloader, project_config.DEVICE)
    
    preds_default = (benchmark_probs >= 0.5).astype(int)
    p_default, r_default, f1_default, _ = precision_recall_fscore_support(benchmark_labels, preds_default, average='binary', zero_division=0)
    
    preds_tuned = (benchmark_probs >= best_threshold).astype(int)
    p_tuned, r_tuned, f1_tuned, _ = precision_recall_fscore_support(benchmark_labels, preds_tuned, average='binary', zero_division=0)
    
    print("\n" + "="*50)
    print("--- FINAL BENCHMARK RESULTS (Threshold Comparison) ---")
    print("="*50)
    print(f"Default Threshold (0.50):")
    print(f"  - F1: {f1_default:.4f}, Precision: {p_default:.4f}, Recall: {r_default:.4f}")
    print(f"TUNED Threshold ({best_threshold:.2f}):")
    print(f"  - F1: {f1_tuned:.4f}, Precision: {p_tuned:.4f}, Recall: {r_tuned:.4f}")
    print("="*50)

    if f1_tuned > f1_default:
        print(f"\nSUCCESS: Threshold tuning improved the F1-score by {f1_tuned - f1_default:.4f}!")
    else:
        print("\nNOTE: Threshold tuning did not improve the F1-score on the benchmark set.")

if __name__ == '__main__':
    try:
        feature_engineering.initialize_feature_engineering()
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()