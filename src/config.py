# src/config.py

import pathlib
import torch

# ------------------------------------------------
# Directory Paths
# ------------------------------------------------
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# ------------------------------------------------
# EXPERIMENT CONFIGURATION
# ------------------------------------------------

RUN_OUTPUT_DIR = MODEL_DIR / "SOTA_model_AraELECTRA_9_features_OriginalData_Tuned_v4"
SAVED_MODEL_DIR = RUN_OUTPUT_DIR / "final_model_with_features"
SCALER_PATH = RUN_OUTPUT_DIR / "scaler_SOTA_AraELECTRA_9_features_v4.joblib"

PRE_TRAINED_MODEL_NAME = "aubmindlab/araelectra-base-discriminator"
PERPLEXITY_MODEL_NAME = "aubmindlab/aragpt2-base"

# ------------------------------------------------
# DATA SETTINGS
# ------------------------------------------------
DATASET_CSV = "data/arabic_training_v4.csv"
TEXT_COLUMN_NAME = "artical_text"
LABEL_COLUMN_NAME = "type"

# ------------------------------------------------
# FEATURE SETTINGS
# ------------------------------------------------
NUM_EXPLICIT_FEATURES = 9
EXPLICIT_FEATURE_COLUMNS = [
    "word_count",
    "sentence_count",
    "avg_sentence_length",
    "sentence_length_std",
    "perplexity",
    "stop_word_ratio",
    "repetition_score_3gram",
    "hapax_ratio",
    "mtld",
  
]
 #"attribution_density",
 #"source_marker_density",
# ------------------------------------------------
# LABEL SETTINGS
# ------------------------------------------------
LABEL_MAP = {"human": 0, "ai": 1}
ID_TO_LABEL = {0: "human", 1: "ai"}
NUM_LABELS = 2

# ------------------------------------------------
# TRAINING HYPERPARAMETERS
# ------------------------------------------------
MAX_LENGTH = 256
BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 1
NUM_TRAIN_EPOCHS = 8

ENCODER_LR             = 1.1e-05   
HEAD_LR                = 4.36e-04  
WEIGHT_DECAY           = 2.5e-05  
#label_smoothing_factor = 0.025381    

# ------------------------------------------------
# DEVICE
# ------------------------------------------------
DEVICE = torch.device("cpu")

# ------------------------------------------------
# OTHER SETTINGS
# ------------------------------------------------
RANDOM_SEED = 42
TEST_SPLIT_SIZE = 0.20
VALIDATION_SPLIT_SIZE = 0.20

print(f"Device: {DEVICE}")
print(f"Model: {PRE_TRAINED_MODEL_NAME}")
print(f"Features: {NUM_EXPLICIT_FEATURES}")
print(f"Epochs: {NUM_TRAIN_EPOCHS}")
print(f"Max length: {MAX_LENGTH}")