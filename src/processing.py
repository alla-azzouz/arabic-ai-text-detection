# src/processing.py

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from . import config

try:
    from pyarabic.araby import normalize_ligature, strip_tashkeel, strip_tatweel, normalize_alef
except ImportError:
    print("Warning: 'pyarabic' library not found. Preprocessing will be limited.")
    # Define dummy functions if pyarabic is not installed
    def normalize_ligature(text): return text
    def strip_tashkeel(text): return text
    def strip_tatweel(text): return text
    def normalize_alef(text): return text

# --- Preprocessing Functions ---

def heavy_preprocess_arabic_text(text: str) -> str:
    """Applies aggressive preprocessing, removing punctuation, numbers, etc."""
    if not isinstance(text, str): return "" 
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = normalize_ligature(text)    
    text = normalize_alef(text)      
    text = strip_tashkeel(text)
    text = strip_tatweel(text)
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF" u"\U00002702-\U000027B0" u"\U000024C2-\U0001F251" 
        u"\U0001F900-\U0001F9FF" u"\U0001FA70-\U0001FAFF" u"\U00002600-\U000026FF"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    punctuations_list = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ''' + r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
    translator = str.maketrans('', '', punctuations_list)
    text = text.translate(translator)
    text = re.sub(r"[٠١٢٣٤٥٦٧٨٩0-9]+", " ", text) 
    text = re.sub(r"[a-zA-Z]+", " ", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def light_preprocess_for_llm(text: str) -> str:
    """Applies only essential normalization for the LLM, preserving signals."""
    if not isinstance(text, str): return "" 
    text = normalize_ligature(text)    
    text = normalize_alef(text)      
    text = strip_tashkeel(text)
    text = strip_tatweel(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Data Loading and Splitting ---

def load_and_prepare_data_for_hybrid_model(csv_path=config.DATASET_CSV,
                                         text_column_name="artical_text", 
                                         label_column_name="type",
                                         use_light_preprocessing=True):
    """
    Loads data and prepares two versions of text for the hybrid model.
    NOTE: This function is primarily used by run_feature_engineering.py.
    """
    print(f"Attempting to load dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    # This function's logic is mostly superseded by the separate run_feature_engineering.py script,
    # but is kept here for reference or potential direct use.
    return df


def split_data(df: pd.DataFrame,
               test_size: float = config.TEST_SPLIT_SIZE,
               validation_size: float = config.VALIDATION_SPLIT_SIZE,
               random_state: int = config.RANDOM_SEED,
               stratify_col: str = 'label_id'):
    """
    Splits the DataFrame into training, validation, and test sets using rich stratification.
    Stratifies by label, AI model (if present), and text length to ensure representative splits.
    """
    assert 0 < test_size < 1 and 0 < validation_size < 1 and (test_size + validation_size) < 1, "Invalid split sizes."

    # --- Build a richer stratification key ---
    # 1. Start with the main label (human/ai)
    strat_key = df[stratify_col].astype(str)
    
    # 2. Add the AI model type if the column exists in the dataframe
    if "model" in df.columns:

        strat_key = strat_key + "_model-" + df["model"].astype(str).fillna("N/A")
    
    # 3. Add text length buckets to stratify by short/medium/long articles
   
    try:
        len_buckets = pd.qcut(df["text"].str.split().apply(len), q=5, labels=False, duplicates='drop').astype(str)
        strat_key = strat_key + "_len-" + len_buckets
    except ValueError as e:
        print(f"Warning: Could not create length buckets for stratification. Stratifying without length. Error: {e}")

    # --- Perform the first split (train+val vs. test) ---
    try:
        df_temp, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=strat_key
        )
    except ValueError:
        print("Warning: Rich stratification failed (likely due to small data groups). Falling back to simple stratification.")
        df_temp, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[stratify_col])

    # --- Perform the second split (train vs. val) ---

    val_relative_size = validation_size / (1.0 - test_size)
    
    strat_key_temp = df_temp[stratify_col].astype(str)
    if "model" in df_temp.columns:
        strat_key_temp = strat_key_temp + "_model-" + df_temp["model"].astype(str).fillna("N/A")
    try:
        len_buckets_temp = pd.qcut(df_temp["text"].str.split().apply(len), q=5, labels=False, duplicates='drop').astype(str)
        strat_key_temp = strat_key_temp + "_len-" + len_buckets_temp
    except ValueError as e:
        print(f"Warning: Could not create length buckets for val split. Stratifying without length. Error: {e}")

    try:
        train_df, val_df = train_test_split(
            df_temp,
            test_size=val_relative_size,
            random_state=random_state,
            stratify=strat_key_temp
        )
    except ValueError:
        print("Warning: Rich stratification failed for val split. Falling back to simple stratification.")
        train_df, val_df = train_test_split(df_temp, test_size=val_relative_size, random_state=random_state, stratify=df_temp[stratify_col])

    print(f"Data split successfully using rich stratification. Sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df
# === END: NEW, ROBUST SPLIT_DATA FUNCTION ===


# --- Tokenizer Loading Function ---
def get_tokenizer(tokenizer_name=config.PRE_TRAINED_MODEL_NAME):
    """
    Loads and returns a tokenizer from Hugging Face.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"Tokenizer '{tokenizer_name}' loaded successfully.")
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer '{tokenizer_name}': {e}")
        return None