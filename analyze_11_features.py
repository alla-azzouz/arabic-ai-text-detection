# analyze_11_features.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import sys

# --- CONFIGURATION FOR THIS SPECIFIC ANALYSIS ---

DATA_FILE_PATH = "data/dataset_with_features_v4.parquet"

# The full list of 11 features to plot.
FEATURE_COLUMNS_TO_PLOT = [
    "word_count", "sentence_count", "avg_sentence_length", "sentence_length_std",
    "perplexity", "stop_word_ratio", "repetition_score_3gram",
    "hapax_ratio", "mtld",
    "attribution_density", 
    "source_marker_density"
]

# The name of the output image file.
OUTPUT_FILENAME = "feature_distributions_11_features.png"

# ===============================================

def plot_feature_distributions():
    """
    Loads the specified dataset and plots the distribution of the 11 features,
    separated by the 'human' and 'ai' labels.
    """
    print(f"--- Loading data from: {DATA_FILE_PATH} ---")
    try:
        df = pd.read_parquet(DATA_FILE_PATH)
    except FileNotFoundError:
        print(f"ERROR: The data file was not found at '{DATA_FILE_PATH}'.")
        print("Please ensure you have generated the 11-feature dataset and that the path is correct.")
        sys.exit(1) # Exit the script if the file is not found

    # Check if all required columns are in the dataframe
    required_cols = ['label_id'] + FEATURE_COLUMNS_TO_PLOT
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"ERROR: The data file is missing the following required columns: {missing_cols}")
        print("This likely means you are pointing to a 9-feature dataset. Please regenerate the 11-feature data file.")
        sys.exit(1)

    # Map the numeric label_id back to descriptive names for the plot legend
    df['label_name'] = df['label_id'].map({0: 'Human', 1: 'AI'})

    # --- Plotting Setup ---
    num_features = len(FEATURE_COLUMNS_TO_PLOT)
    # Arrange plots in a 4x3 grid for 11 features
    n_cols = 3
    n_rows = math.ceil(num_features / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    print(f"--- Generating {num_features} distribution plots... ---")
    for i, feature in enumerate(FEATURE_COLUMNS_TO_PLOT):
        ax = axes[i]
        
        sns.kdeplot(data=df, x=feature, hue='label_name', fill=True, 
                    palette={'Human': 'blue', 'AI': 'red'}, ax=ax, warn_singular=False)
        
        ax.set_title(f'Distribution of "{feature}"', fontsize=12)
        ax.set_xlabel('')
        ax.set_ylabel('Density')
    
    # Hide the last unused subplot in the 4x3 grid
    for j in range(num_features, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(pad=3.0)
    
    # --- Save the final figure ---
    plt.savefig(OUTPUT_FILENAME, dpi=300)
    print(f"\n--- Plots saved successfully to: {OUTPUT_FILENAME} ---")
    
    # Optionally, display the plot
    plt.show()

if __name__ == '__main__':
    plot_feature_distributions()