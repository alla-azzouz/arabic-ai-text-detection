Arabic AI-Generated Text Detection: A Hybrid Transformer and Linguistic Feature Framework
This repository contains the code, data, and resources for the paper:

"Hybrid Framework for Robust AI-Generated Arabic Text Detection with Cross-Domain Generalization and Human-Machine Comparative Analysis"

Overview
This work proposes a hybrid framework for detecting AI-generated Arabic text by combining Arabic-specific pretrained transformer encoders (AraELECTRA and AraBERT) with explicit Arabic linguistic features. Four model configurations are evaluated across two feature sets (9-feature and 11-feature) on two independent Arabic benchmark datasets.
Repository Structure
├── src/
│   ├── config.py              # Model and training configuration
│   ├── model.py               # Hybrid model architecture
│   ├── train.py               # Training script
│   ├── feature_engineering.py # Arabic linguistic feature extraction
│   ├── processing.py          # Data preprocessing
│   └── __init__.py
├── data/
│   ├── arabic_training_v4.csv                 # Final training dataset (8,041 samples)
│   ├── dataset_with_features_v4.parquet       # Training dataset with precomputed features
│   ├── airabic_holdout_test_v3.csv            # AIRABIC benchmark holdout (600 samples)
│   ├── airabic_with_features.parquet          # AIRABIC precomputed features
│   ├── alhd_holdout_test.csv                  # ALHD benchmark holdout (18,268 samples)
│   ├── alhd_with_features.parquet             # ALHD precomputed features
│   └── human_eval_texts.csv                   # Human evaluation texts (14 samples)
├── optuna_results_9features.csv                                # AraELECTRA 9feat Optuna results
├── optuna_results_araelectra-base-discriminator_11features.csv # AraELECTRA 11feat Optuna results
├── optuna_results_bert-base-arabertv2_9features.csv            # AraBERT 9feat Optuna results
├── optuna_arabertv2_11features.csv                             # AraBERT 11feat Optuna results
├── evaluate_on_benchmark.py   # AIRABIC benchmark evaluation
├── evaluate_on_alhd.py        # ALHD benchmark evaluation
├── human_evaluation.py        # Human expert evaluation
├── build_v4_dataset.py        # Dataset v4 construction script
├── add_11feat_to_parquet.py   # Add 11 features to parquet
├── tune_hyperparameters.py    # Optuna hyperparameter optimization
├── tune_threshold.py          # Decision threshold analysis
├── analyze_features.py        # 9-feature distribution analysis
├── analyze_11_features.py     # 11-feature distribution analysis
└── requirements.txt
Dataset
The training dataset contains 8,041 Arabic text samples after deduplication:

Human-written: 4,091 samples from diverse Arabic sources
AI-generated: 3,950 samples from 6 language models including Arabic-native models Jais and ALLaM
Balance ratio: 1.035 (near-perfectly balanced)
Sources: Modern lifestyle Arabic, classical Islamic Arabic, hard news, social media

The dataset was split using composite stratification (source + word count) with a 60/20/20 train/validation/test ratio.
Model Weights
Pre-trained model weights for all four configurations are available on HuggingFace:

AraELECTRA 9feat: https://huggingface.co/alla4a/araelectra-9feat-arabic-ai-detection
AraELECTRA 11feat: https://huggingface.co/alla4a/araelectra-11feat-arabic-ai-detection
AraBERT 9feat: https://huggingface.co/alla4a/arabert-9feat-arabic-ai-detection
AraBERT 11feat: https://huggingface.co/alla4a/arabert-11feat-arabic-ai-detection

Download and place each model folder under the models/ directory:
models/
├── SOTA_model_AraELECTRA_9_features_OriginalData_Tuned_v4/
├── SOTA_model_AraELECTRA_11_features_Augmented_Tuned_v4/
├── SOTA_model_AraBERT_9_features_OriginalData_Tuned_v4/
└── SOTA_model_AraBERT_11_features_OriginalData_Tuned_v4/
Installation
bashgit clone https://github.com/alla-azzouz/ai-text-detector.git
cd ai-text-detector
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
Usage
Training
Update src/config.py with your desired model configuration then run:
bashpython -m src.train
Evaluation on AIRABIC Benchmark
bashpython evaluate_on_benchmark.py
Evaluation on ALHD Benchmark
bashpython evaluate_on_alhd.py
Human Evaluation
bashpython human_evaluation.py
Hyperparameter Optimization
bashpython tune_hyperparameters.py
Feature Distribution Analysis
bashpython analyze_features.py        # 9 features
python analyze_11_features.py     # 11 features
Model Configuration
To switch between model configurations, update src/config.py:
ModelRUN_OUTPUT_DIRPRE_TRAINED_MODEL_NAMENUM_EXPLICIT_FEATURESAraELECTRA 9featSOTA_model_AraELECTRA_9_features_OriginalData_Tuned_v4aubmindlab/araelectra-base-discriminator9AraELECTRA 11featSOTA_model_AraELECTRA_11_features_Augmented_Tuned_v4aubmindlab/araelectra-base-discriminator11AraBERT 9featSOTA_model_AraBERT_9_features_OriginalData_Tuned_v4aubmindlab/bert-base-arabertv29AraBERT 11featSOTA_model_AraBERT_11_features_OriginalData_Tuned_v4aubmindlab/bert-base-arabertv211
Linguistic Features
9-Feature Set
FeatureDescriptionword_countTotal number of wordssentence_countTotal number of sentencesavg_sentence_lengthAverage words per sentencesentence_length_stdStandard deviation of sentence lengthsperplexityAraGPT2-based Arabic perplexity scorestop_word_ratioRatio of Arabic stop wordsrepetition_score_3gram3-gram uniqueness scorehapax_ratioRatio of words appearing only oncemtldMeasure of Textual Lexical Diversity
11-Feature Set (9 features + 2 additional)
FeatureDescriptionattribution_densityDensity of attribution markerssource_marker_densityDensity of source citation markers
Citation

