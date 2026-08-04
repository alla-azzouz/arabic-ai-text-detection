import pandas as pd
from src import processing, feature_engineering
from tqdm import tqdm
tqdm.pandas()

# Load existing parquet
df = pd.read_parquet("data/dataset_with_features_v4.parquet")
print(f"Current parquet: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Initialize feature engineering
feature_engineering.initialize_feature_engineering()

# Calculate only the 2 missing features
print("\nCalculating attribution_density and source_marker_density...")
print("This will take ~30-40 minutes for 8,041 samples...")

def calc_two_features(text):
    import re
    ATTRIBUTION_VERBS = [
        "قال", "أوضح", "صرح", "وفقًا لـ", "نقلاً عن", "ذكرت",
        "أضاف", "أشار", "أكد", "اعتبر", "بحسب"
    ]
    SOURCE_MARKERS = [
        "رويترز", "واس", "وكالة الأنباء", "بيان", "المصدر",
        "سي ان ان", "بي بي سي", "سكاي نيوز", "العربية"
    ]
    ATTRIBUTION_REGEX = re.compile(r'\b(' + '|'.join(ATTRIBUTION_VERBS) + r')\b', re.IGNORECASE)
    SOURCE_REGEX = re.compile(r'(' + '|'.join(SOURCE_MARKERS) + r')', re.IGNORECASE)

    words = text.split()
    word_count = max(len(words), 1)
    attribution = len(ATTRIBUTION_REGEX.findall(text)) / word_count * 1000
    source = len(SOURCE_REGEX.findall(text)) / word_count * 1000
    return attribution, source

results = df["text"].progress_apply(calc_two_features)
df["attribution_density"]   = [r[0] for r in results]
df["source_marker_density"] = [r[1] for r in results]

print(f"\nNew parquet shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Save as new parquet — keep v4 name
df.to_parquet("data/dataset_with_features_v4.parquet", index=False)
print("Saved: data/dataset_with_features_v4.parquet ✅")