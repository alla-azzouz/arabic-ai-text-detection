# analyze_features.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# ===== Project modules =====
from src import config as project_config

# --- Configuration ---
# The processed data file to analyze
DATA_FILE_PATH = project_config.DATA_DIR / "dataset_with_features_v4.parquet"
# The columns we want to plot
FEATURE_COLUMNS_TO_PLOT = project_config.EXPLICIT_FEATURE_COLUMNS

def plot_feature_distributions():
    """
    Loads the processed dataset and plots the distribution of each feature,
    separated by the 'human' and 'ai' labels.
    """
    print(f"--- Loading data from: {DATA_FILE_PATH} ---")
    try:
        df = pd.read_parquet(DATA_FILE_PATH)
    except FileNotFoundError:
        print(f"ERROR: The data file was not found. Please run 'run_feature_engineering.py' first.")
        return

    # Map the numeric label_id back to descriptive names for the plot legend
    df['label_name'] = df['label_id'].map({0: 'Human', 1: 'AI'})

    # --- Plotting Setup ---
    num_features = len(FEATURE_COLUMNS_TO_PLOT)
    # Arrange plots in a grid (e.g., 3x3 for 9 features)
    n_cols = 3
    n_rows = math.ceil(num_features / n_cols)
    
    # Create the figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    print(f"--- Generating {num_features} distribution plots... ---")
    for i, feature in enumerate(FEATURE_COLUMNS_TO_PLOT):
        ax = axes[i]
        
        # Use seaborn's Kernel Density Estimate (KDE) plot for smooth distributions
        sns.kdeplot(data=df, x=feature, hue='label_name', fill=True, 
                    palette={'Human': 'blue', 'AI': 'red'}, ax=ax)
        
        ax.set_title(f'Distribution of "{feature}"', fontsize=12)
        ax.set_xlabel('') # Keep it clean
        ax.set_ylabel('Density')
    

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Adjust layout to prevent titles from overlapping
    plt.tight_layout(pad=3.0)
    
    # --- Save the final figure ---
    output_filename = "feature_distributions.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\n--- Plots saved successfully to: {output_filename} ---")
    
    # Display the plot
    plt.show()

if __name__ == '__main__':
    plot_feature_distributions()