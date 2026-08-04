# src/train.py

import math
import random
import dataclasses
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import joblib

from sklearn.preprocessing import StandardScaler
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
)

from datasets import Dataset
from torch.optim import AdamW

from transformers import (
    TrainingArguments,
    Trainer,
    EvalPrediction,
    EarlyStoppingCallback,
    get_scheduler,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from . import config as project_config
from . import processing
from . import model


def set_seed(seed_value: int = project_config.RANDOM_SEED) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def compute_metrics(p: EvalPrediction) -> dict:
    """Compute evaluation metrics from Trainer predictions."""
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids

    preds = np.argmax(logits, axis=1)
    probs_ai = softmax(logits, axis=1)[:, 1]

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )
    acc = accuracy_score(labels, preds)

    try:
        roc_auc = roc_auc_score(labels, probs_ai)
    except ValueError:
        roc_auc = 0.5

    try:
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    except ValueError:
        tn, fp, fn, tp, specificity = 0, 0, 0, 0, 0.0

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "specificity": specificity,
        "true_positives_tp": int(tp),
        "false_positives_fp": int(fp),
        "false_negatives_fn": int(fn),
        "true_negatives_tn": int(tn),
    }


def create_dataset(df: pd.DataFrame, scaled_features: np.ndarray) -> Dataset:
    """Create Hugging Face Dataset from dataframe and explicit features."""
    return Dataset.from_dict(
        {
            "text": df["text"].tolist(),
            "labels": df["label_id"].astype(int).tolist(),
            "explicit_features": scaled_features.astype("float32").tolist(),
        }
    )


@dataclasses.dataclass
class CustomDataCollatorWithFeatures:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[dict[str, Any]]) -> dict[str, Any]:
        labels = [f.pop("labels") for f in features] if "labels" in features[0] else None
        explicit_features_list = (
            [f.pop("explicit_features") for f in features]
            if "explicit_features" in features[0]
            else None
        )

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        if labels is not None:
            batch["labels"] = torch.tensor(labels, dtype=torch.long)

        if explicit_features_list is not None:
            batch["explicit_features"] = torch.tensor(
                explicit_features_list,
                dtype=torch.float32,
            )

        return batch


def main_train() -> None:
    set_seed()
    project_config.RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device for training: {project_config.DEVICE}")

    # ---------------------------------------------------------
    # 1. Load pre-calculated dataset with features
    # ---------------------------------------------------------
    print("\n--- 1. Loading PRE-CALCULATED Data with Features ---")
    parquet_v3 = project_config.DATA_DIR / "dataset_with_features_v4.parquet"
    parquet_default = project_config.DATA_DIR / "dataset_with_features.parquet"

    if parquet_v3.exists():
        processed_data_path = parquet_v3
    else:
        processed_data_path = parquet_default
        print(f"WARNING: dataset_with_features_v3.parquet not found. Using {processed_data_path}")

    print(f"Loading: {processed_data_path}")
    full_df = pd.read_parquet(processed_data_path)

    print(
        f"Total: {len(full_df)} | "
        f"Human: {len(full_df[full_df['label_id'] == 0])} | "
        f"AI: {len(full_df[full_df['label_id'] == 1])}"
    )

    train_df, val_df, test_df = processing.split_data(full_df)

    # ---------------------------------------------------------
    # 2. Scale explicit features
    # ---------------------------------------------------------
    print("\n--- 2. Scaling Explicit Features ---")
    feat_cols = project_config.EXPLICIT_FEATURE_COLUMNS

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_df[feat_cols])
    X_val_scaled = scaler.transform(val_df[feat_cols])
    X_test_scaled = scaler.transform(test_df[feat_cols])

    joblib.dump(scaler, project_config.SCALER_PATH)
    print(f"Scaler saved to: {project_config.SCALER_PATH}")

    # ---------------------------------------------------------
    # 3. Create HF datasets and tokenize
    # ---------------------------------------------------------
    print("\n--- 3. Creating HF Datasets ---")
    tokenizer = processing.get_tokenizer(project_config.PRE_TRAINED_MODEL_NAME)

    raw_train = create_dataset(train_df, X_train_scaled)
    raw_val = create_dataset(val_df, X_val_scaled)
    raw_test = create_dataset(test_df, X_test_scaled)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=project_config.MAX_LENGTH,
        )

    train_ds = raw_train.map(tokenize_fn, batched=True, remove_columns=["text"])
    val_ds = raw_val.map(tokenize_fn, batched=True, remove_columns=["text"])
    test_ds = raw_test.map(tokenize_fn, batched=True, remove_columns=["text"])

    # ---------------------------------------------------------
    # 4. Load model
    # ---------------------------------------------------------
    print("\n--- 4. Loading ADVANCED CUSTOM Model ---")
    net = model.load_advanced_model(
        model_name_or_path=project_config.PRE_TRAINED_MODEL_NAME,
        num_labels=project_config.NUM_LABELS,
        id2label=project_config.ID_TO_LABEL,
        label2id=project_config.LABEL_MAP,
        num_explicit_features=project_config.NUM_EXPLICIT_FEATURES,
    )
    net.to(project_config.DEVICE)

    print("Configured device:", project_config.DEVICE)
    print("Model device:", next(net.parameters()).device)

    # ---------------------------------------------------------
    # 5. Training arguments
    # ---------------------------------------------------------
    print("\n--- 5. Defining Training Arguments ---")
    training_args = TrainingArguments(
        output_dir=str(project_config.SAVED_MODEL_DIR),
        num_train_epochs=project_config.NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=project_config.BATCH_SIZE,
        per_device_eval_batch_size=project_config.BATCH_SIZE,
        gradient_accumulation_steps=project_config.GRADIENT_ACCUMULATION,
        weight_decay=project_config.WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=project_config.RANDOM_SEED,
        remove_unused_columns=False,
        label_smoothing_factor=0.025381 , # AraELECTRA Optuna best,
        save_total_limit=2,
        no_cuda=True,
        use_mps_device=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
    )

    # ---------------------------------------------------------
    # 6. Optimizer + scheduler
    # ---------------------------------------------------------
    print("\n--- 6. Defining Custom Optimizer and Scheduler ---")

    head_param_prefixes = ("feature_tower.", "classifier.")

    encoder_params = [
        p
        for n, p in net.named_parameters()
        if p.requires_grad and not n.startswith(head_param_prefixes)
    ]

    head_params = [
        p
        for n, p in net.named_parameters()
        if p.requires_grad and n.startswith(head_param_prefixes)
    ]

    print(f"Encoder params: {sum(p.numel() for p in encoder_params):,}")
    print(f"Head params   : {sum(p.numel() for p in head_params):,}")

    if len(head_params) == 0:
        raise RuntimeError("Head parameter group is empty. Check parameter grouping logic.")

    optimizer = AdamW(
        [
            {
                "params": encoder_params,
                "lr": project_config.ENCODER_LR,
                "weight_decay": training_args.weight_decay,
            },
            {
                "params": head_params,
                "lr": project_config.HEAD_LR,
                "weight_decay": 0.0,
            },
        ]
    )

    updates_per_epoch = math.ceil(
        len(train_ds)
        / (
            training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
        )
    )
    num_training_steps = updates_per_epoch * int(training_args.num_train_epochs)
    num_warmup_steps = max(1, int(0.08 * num_training_steps))

    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # ---------------------------------------------------------
    # 7. Trainer
    # ---------------------------------------------------------
    print("\n--- 7. Initializing Trainer ---")
    collator = CustomDataCollatorWithFeatures(tokenizer=tokenizer)

    trainer = Trainer(
        model=net,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        optimizers=(optimizer, scheduler),
    )

    # ---------------------------------------------------------
    # 8. Train
    # ---------------------------------------------------------
    print("\n--- 8. Starting Model Training ---")
    trainer.train()

    # ---------------------------------------------------------
    # 9. Evaluate best model on test set
    # ---------------------------------------------------------
    print("\n--- 9. Evaluating Best Model on Test Set ---")
    test_pred = trainer.predict(test_ds)
    test_metrics = compute_metrics(test_pred)

    print(f"Final evaluation on test set (best model): {test_metrics}")
    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)

    # ---------------------------------------------------------
    # 10. Save model
    # ---------------------------------------------------------
    print("\n--- 10. Saving Fine-tuned Model ---")
    project_config.SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    trainer.save_model()

    complete_state_dict = trainer.model.state_dict()
    state_dict_path = project_config.SAVED_MODEL_DIR / "best_model_state_dict.pt"
    torch.save(complete_state_dict, str(state_dict_path))

    print(f"Fine-tuned model saved to: {project_config.SAVED_MODEL_DIR}")
    print(f"Complete state dict saved to: {state_dict_path}")

    enc_keys = [k for k in complete_state_dict.keys() if "transformer.encoder.layer.6" in k]
    if enc_keys:
        w = complete_state_dict[enc_keys[0]].float()
        std = w.std().item()
        mean = w.mean().item()
        print(f"Transformer layer 6 — std={std:.6f}, mean={mean:.6f}")

    print("\n--- Training Script Completed ---")


if __name__ == "__main__":
    print("Starting training process for hybrid model...")
    main_train()