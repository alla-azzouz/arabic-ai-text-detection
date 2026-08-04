# tune_hyperparameters.py

import torch
import numpy as np
import pandas as pd
import joblib
import math
import optuna
import csv
import time
import shutil
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW

# ===== HF Transformers & Optuna Integration =====
from transformers import (
    TrainingArguments,
    Trainer,
    TrainerCallback, # Import the base class
    EarlyStoppingCallback,
    get_scheduler,
)
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# ===== Project modules =====
from src import config as project_config
from src import processing
from src import model
from src.train import (
    set_seed,
    create_dataset,
    CustomDataCollatorWithFeatures,
    compute_metrics,
)

# ------------------------- TUNING CONFIGURATION -------------------------
N_TRIALS = 50
N_EPOCHS = 4
# ----------------------------------------------------------------------

# === Custom Classes ===
class OptunaTrainer(Trainer):
    def _save_optimizer_and_scheduler(self, output_dir):
        pass

class OptunaPruningCallback(TrainerCallback):
    """Custom pruning callback that works with any version."""
    def __init__(self, trial: optuna.Trial, monitor: str):
        self.trial = trial
        self.monitor = monitor

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and self.monitor in metrics:
            value = metrics[self.monitor]
            self.trial.report(float(value), step=state.global_step)
            if self.trial.should_prune():
                raise optuna.TrialPruned(f"Pruned at step {state.global_step} with {self.monitor}={value}")

def save_trial_to_csv(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    model_name = project_config.PRE_TRAINED_MODEL_NAME.split('/')[-1]
    num_features = project_config.NUM_EXPLICIT_FEATURES
    filename = f"optuna_results_{model_name}_{num_features}features.csv"
    metric_keys = ["f1","precision","recall","specificity","roc_auc","accuracy","loss", "true_positives_tp","false_positives_fp","false_negatives_fn","true_negatives_tn", "runtime_sec","samples_per_sec","threshold"]
    row = {"trial_number": trial.number, "f1_score": trial.value}
    for k in metric_keys:
        row[k] = trial.user_attrs.get(k)
    row.update(trial.params)
    fixed_cols = ["trial_number","f1_score"] + metric_keys
    param_cols = sorted([p for p in trial.params.keys()])
    headers = fixed_cols + param_cols
    file_exists = False
    try:
        with open(filename, "r") as _: file_exists = True
    except FileNotFoundError: pass
    with open(filename, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

# --- Load data once ---
print("--- Loading and preparing data for Optuna study ---")
processed_data_path = project_config.DATA_DIR / "dataset_with_features.parquet"
full_df = pd.read_parquet(processed_data_path)
train_df, val_df, _ = processing.split_data(full_df)
feat_cols = project_config.EXPLICIT_FEATURE_COLUMNS
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_df[feat_cols])
X_val_scaled = scaler.transform(val_df[feat_cols])
tokenizer = processing.get_tokenizer(project_config.PRE_TRAINED_MODEL_NAME)
raw_train = create_dataset(train_df, X_train_scaled)
raw_val = create_dataset(val_df, X_val_scaled)
def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=project_config.MAX_LENGTH)
train_ds = raw_train.map(tokenize_fn, batched=True, remove_columns=["text"])
val_ds = raw_val.map(tokenize_fn, batched=True, remove_columns=["text"])


def objective(trial: optuna.Trial) -> float:
    set_seed(project_config.RANDOM_SEED)
    encoder_lr = trial.suggest_float("encoder_lr", 1e-6, 5e-5, log=True)
    head_lr = trial.suggest_float("head_lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 0.1, log=True)
    label_smoothing = trial.suggest_float("label_smoothing_factor", 0.0, 0.2)
    
    print(f"\n\n--- Starting Trial #{trial.number} ---")
    print(f"  - Params: EncLR={encoder_lr:.2e}, HeadLR={head_lr:.2e}, WD={weight_decay:.2e}, LS={label_smoothing:.3f}")
    output_dir = f"./optuna_trials/trial_{trial.number}"
    
    net = model.load_advanced_model(
        model_name_or_path=project_config.PRE_TRAINED_MODEL_NAME,
        num_labels=project_config.NUM_LABELS,
        id2label=project_config.ID_TO_LABEL,
        label2id=project_config.LABEL_MAP,
        num_explicit_features=project_config.NUM_EXPLICIT_FEATURES,
    ).to(project_config.DEVICE)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=N_EPOCHS,
        per_device_train_batch_size=project_config.BATCH_SIZE,
        per_device_eval_batch_size=project_config.BATCH_SIZE,
        gradient_accumulation_steps=project_config.GRADIENT_ACCUMULATION,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",
        seed=project_config.RANDOM_SEED,
        remove_unused_columns=False,
        label_smoothing_factor=label_smoothing,
    )

    head_names = set()
    for submod_name in ["feature_tower", "classifier"]:
        submod = getattr(net, submod_name)
        for n, _ in submod.named_parameters():
            head_names.add(f"{submod_name}.{n}")
    encoder_params = [p for n, p in net.named_parameters() if n not in head_names and p.requires_grad]
    head_params = [p for n, p in net.named_parameters() if n in head_names and p.requires_grad]
    optimizer_grouped_parameters = [{"params": encoder_params, "lr": encoder_lr, "weight_decay": weight_decay}, {"params": head_params, "lr": head_lr, "weight_decay": 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters)
    updates_per_epoch = math.ceil(len(train_ds) / (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps))
    num_training_steps = updates_per_epoch * int(training_args.num_train_epochs)
    num_warmup_steps = max(1, int(0.08 * num_training_steps))
    scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    
    collator = CustomDataCollatorWithFeatures(tokenizer=tokenizer)
    pruning_cb = OptunaPruningCallback(trial, monitor="eval_f1")

    trainer = OptunaTrainer(
        model=net, args=training_args, train_dataset=train_ds, eval_dataset=val_ds,
        tokenizer=tokenizer, data_collator=collator, compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2), pruning_cb],
        optimizers=(optimizer, scheduler),
    )

    t0 = time.perf_counter()
    trainer.train()
    runtime = time.perf_counter() - t0
    eval_metrics = trainer.evaluate()
    best_f1_score = float(trainer.state.best_metric) if trainer.state.best_metric is not None else float(eval_metrics.get("eval_f1", 0.0))

    for key in eval_metrics:
        if key.startswith("eval_"):
            trial.set_user_attr(key[5:], eval_metrics[key])
    trial.set_user_attr("runtime_sec", runtime)
    trial.set_user_attr("samples_per_sec", ((len(train_ds) + len(val_ds)) / runtime) if runtime > 0 else None)
    trial.set_user_attr("threshold", 0.5)

    print(f"--- Trial #{trial.number} Finished ---")
    print(f"  - Final Validation F1: {best_f1_score:.4f} | Runtime: {runtime:.2f}s")
    
    try:
        shutil.rmtree(output_dir)
        print(f"  - Cleaned up directory: {output_dir}")
    except OSError as e:
        print(f"  - Error cleaning up directory {output_dir}: {e.strerror}")
    
    return float(best_f1_score)


if __name__ == "__main__":
    print(f"Starting Optuna hyperparameter search for {N_TRIALS} trials...")
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(multivariate=True, group=True, n_startup_trials=10),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=0)
    )
    study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True, callbacks=[save_trial_to_csv])

    print("\n\n" + "="*50)
    print("--- OPTIMIZATION FINISHED ---")
    print("="*50)
    print(f"  Number of finished trials: {len(study.trials)}")
    if study.best_trial:
        print("\n--- Best Trial ---")
        best_trial = study.best_trial
        print(f"  - Value (Best F1 Score): {best_trial.value:.4f}")
        print("  - Best Hyperparameters:")
        for key, value in best_trial.params.items():
            print(f"    - {key}: {value}")
    else:
        print("No successful trials were completed.")
    print("\n--- Next Steps ---")
    print("1. A CSV file with all trial results has been saved in this directory.")
    print("2. Update your `src/config.py` and `src/train.py` with the 'Best Hyperparameters' found.")
    print("3. Run the main `src/train.py` script one final time to train your model to completion.")