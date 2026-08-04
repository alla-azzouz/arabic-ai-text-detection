# src/feature_engineering.py

import re
import numpy as np
import pysbd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter
from lexical_diversity import lex_div as ld
from . import config as project_config

# --- Global variables ---
ARABIC_SEGMENTER = None
PPL_MODEL = None
PPL_TOKENIZER = None

# --- Lists and Regex for feature detection ---

ATTRIBUTION_VERBS = [
    "قال", "أوضح", "صرح", "وفقًا لـ", "نقلاً عن", "ذكرت", "أضاف", "أشار", "أكد", "اعتبر", "بحسب"
]
SOURCE_MARKERS = [
    "رويترز", "واس", "وكالة الأنباء", "بيان", "المصدر", "سي ان ان", "بي بي سي", "سكاي نيوز", "العربية"
]
ATTRIBUTION_REGEX = re.compile(r'\b(' + '|'.join(ATTRIBUTION_VERBS) + r')\b', re.IGNORECASE)
SOURCE_REGEX = re.compile(r'(' + '|'.join(SOURCE_MARKERS) + r')', re.IGNORECASE)
# ========================================================

ARABIC_STOP_WORDS = set([
    "من", "في", "على", "إلى", "عن", "هو", "هي", "هم", "هن", "هذا", "هذه", "ذلك", "تلك", "كان", "يكون",
    "إن", "أن", "لكن", "كل", "بعض", "قد", "سوف", "تم", "التي", "الذي", "الذين", "مع", "به", "له", "اذا",
    "كما", "أو", "و", "ف", "ثم", "حتى", "أي", "ما", "ماذا", "لماذا", "كيف", "متى", "اين", "ايضا"
])

def initialize_feature_engineering():
    """Initializes all necessary models and tools for feature engineering."""
    global ARABIC_SEGMENTER, PPL_MODEL, PPL_TOKENIZER
    if ARABIC_SEGMENTER is None:
        print("Initializing PySBD sentence segmenter for Arabic...")
        try:
            ARABIC_SEGMENTER = pysbd.Segmenter(language="ar", clean=False)
            print("PySBD segmenter initialized.")
        except Exception as e:
            print(f"ERROR: Failed to initialize PySBD segmenter: {e}")
    if PPL_MODEL is None and hasattr(project_config, 'PERPLEXITY_MODEL_NAME'):
        print(f"Initializing Perplexity model ({project_config.PERPLEXITY_MODEL_NAME})...")
        try:
            PPL_TOKENIZER = AutoTokenizer.from_pretrained(project_config.PERPLEXITY_MODEL_NAME)
            PPL_MODEL = AutoModelForCausalLM.from_pretrained(project_config.PERPLEXITY_MODEL_NAME)
            PPL_MODEL.to(project_config.DEVICE)
            PPL_MODEL.eval()
            print("Perplexity model initialized.")
        except Exception as e:
            print(f"ERROR: Failed to initialize Perplexity model: {e}")

def calculate_perplexity(text: str) -> float:
    global PPL_MODEL, PPL_TOKENIZER
    if PPL_MODEL is None or PPL_TOKENIZER is None or not text.strip():
        return 0.0
    with torch.no_grad():
        inputs = PPL_TOKENIZER(text, return_tensors="pt", max_length=512, truncation=True)
        input_ids = inputs.input_ids.to(project_config.DEVICE)
        if input_ids.size(1) < 2: return 0.0
        outputs = PPL_MODEL(input_ids, labels=input_ids)
        try:
            ppl = torch.exp(outputs.loss).item()
        except (OverflowError, ValueError):
            ppl = float('inf')
    return ppl if ppl != float('inf') else 100000.0

def calculate_repetition_score(text: str, n: int = 3) -> float:
    words = text.split()
    if len(words) < n: return 1.0
    ngrams = [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]
    if not ngrams: return 1.0
    unique_ngrams = set(ngrams)
    return len(unique_ngrams) / len(ngrams)

def calculate_hapax_ratio(words: list) -> float:
    if not words: return 0.0
    freqs = Counter(words)
    hapaxes = sum(1 for count in freqs.values() if count == 1)
    return hapaxes / len(words)

def calculate_mtld(words: list) -> float:
    if len(words) < 50: return 0.0
    return ld.mtld(words)

def calculate_features(text: str) -> dict:
    """Calculates all 11 explicit features for a given text."""
    global ARABIC_SEGMENTER
    features = {}
    
    lightly_cleaned_text = re.sub(r'\s+', ' ', text).strip()
    words = lightly_cleaned_text.split()
    word_count = len(words)
    
    features["word_count"] = float(word_count)
    
    if ARABIC_SEGMENTER and lightly_cleaned_text:
        sentences = ARABIC_SEGMENTER.segment(lightly_cleaned_text)
        sentence_count = len(sentences)
        sentence_lengths = [len(s.split()) for s in sentences] if sentences else [0]
        features["sentence_count"] = float(sentence_count)
        features["avg_sentence_length"] = np.mean(sentence_lengths) if sentence_lengths else 0.0
        features["sentence_length_std"] = np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0.0
    else:
        features["sentence_count"] = 1.0
        features["avg_sentence_length"] = float(word_count)
        features["sentence_length_std"] = 0.0

    if word_count > 0:
        features["stop_word_ratio"] = sum(1 for word in words if word in ARABIC_STOP_WORDS) / word_count
        features["repetition_score_3gram"] = calculate_repetition_score(lightly_cleaned_text, n=3)
        features["hapax_ratio"] = calculate_hapax_ratio(words)
        features["mtld"] = calculate_mtld(words)
        
        # === THIS IS THE CRITICAL SECTION THAT MUST BE CORRECT ===
        # Calculate density per 1,000 words for stability
        features["attribution_density"] = len(ATTRIBUTION_REGEX.findall(lightly_cleaned_text)) / word_count * 1000
        features["source_marker_density"] = len(SOURCE_REGEX.findall(lightly_cleaned_text)) / word_count * 1000
        # ========================================================
    else: 
        features["stop_word_ratio"], features["repetition_score_3gram"] = 0.0, 1.0
        features["hapax_ratio"], features["mtld"] = 0.0, 0.0
        
        # === MAKE SURE THE NEW FEATURES ARE ZERO FOR EMPTY TEXT ===
        features["attribution_density"] = 0.0
        features["source_marker_density"] = 0.0
        # ========================================================

    features["perplexity"] = np.log1p(calculate_perplexity(lightly_cleaned_text))
    
    for key in features:
        features[key] = float(features[key])
            
    return features