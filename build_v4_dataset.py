import pandas as pd
import numpy as np

# ── Load current training parquet (v3) ───────────────────────
df_train = pd.read_parquet("data/dataset_with_features.parquet")
print(f"Current v3 parquet: {df_train.shape}")
print(f"Human: {len(df_train[df_train['label_id']==0])}")
print(f"AI:    {len(df_train[df_train['label_id']==1])}")

# ── Load ALHD ─────────────────────────────────────────────────
df_alhd = pd.read_csv("data/ALHD_balanced_10percent.csv")
print(f"\nALHD total: {len(df_alhd)}")

# ── Stratified split ALHD ─────────────────────────────────────
# Take 1000 Human + 1000 AI for training
# Keep rest for holdout test
from sklearn.model_selection import train_test_split

alhd_human = df_alhd[df_alhd['label']==0].copy()
alhd_ai    = df_alhd[df_alhd['label']==1].copy()

# Stratify by subcategory for representative sample
human_train, human_test = train_test_split(
    alhd_human, train_size=1000, random_state=42, stratify=alhd_human['subcategory'])
ai_train, ai_test = train_test_split(
    alhd_ai, train_size=1000, random_state=42, stratify=alhd_ai['subcategory'])

print(f"\nALHD for training  : {len(human_train)} Human + {len(ai_train)} AI")
print(f"ALHD for holdout   : {len(human_test)} Human + {len(ai_test)} AI")

# ── Save ALHD holdout test set ────────────────────────────────
alhd_holdout = pd.concat([human_test, ai_test], ignore_index=True)
alhd_holdout = alhd_holdout.sample(frac=1, random_state=42).reset_index(drop=True)
alhd_holdout.to_csv("data/alhd_holdout_test.csv", index=False)
print(f"\nSaved: data/alhd_holdout_test.csv ({len(alhd_holdout)} samples)")

# ── Prepare ALHD training rows for parquet ───────────────────
# Need to add to parquet with same columns as v3
# Parquet columns: text, label_id, word_count, sentence_count, ...


alhd_train = pd.concat([human_train, ai_train], ignore_index=True)
alhd_train_csv = alhd_train[['text', 'label']].copy()
alhd_train_csv.columns = ['artical_text', 'type']
alhd_train_csv['model'] = alhd_train['generator'].values
alhd_train_csv.to_csv("data/alhd_training_samples.csv", index=False)
print(f"Saved: data/alhd_training_samples.csv ({len(alhd_train_csv)} samples)")

# ── Create arabic_training_v4.csv ────────────────────────────
df_v3_csv = pd.read_csv("data/arabic_training_v3.csv")
df_v4 = pd.concat([df_v3_csv, alhd_train_csv], ignore_index=True)
df_v4 = df_v4.drop_duplicates(subset=['artical_text'])
df_v4 = df_v4.sample(frac=1, random_state=42).reset_index(drop=True)
df_v4.to_csv("data/arabic_training_v4.csv", index=False)

print(f"\n=== arabic_training_v4.csv ===")
print(f"Total  : {len(df_v4)}")
print(f"Human  : {len(df_v4[df_v4['type']==0])}")
print(f"AI     : {len(df_v4[df_v4['type']==1])}")
print(f"Balance: {len(df_v4[df_v4['type']==0])/len(df_v4[df_v4['type']==1]):.3f}")
print(f"\nSaved: data/arabic_training_v4.csv")
print("\nNext step: calculate features for ALHD training samples and add to parquet")