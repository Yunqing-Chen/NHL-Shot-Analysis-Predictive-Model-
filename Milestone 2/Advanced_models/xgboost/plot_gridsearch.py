import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_grid_search_from_csv(csv_path, param_name1, param_name2, score_column="mean_test_score"):
    """
    Reads grid search results from a CSV file, aggregates duplicates, and plots a heatmap for two specified parameters.

    Parameters:
        csv_path (str): Path to the grid search results CSV file.
        param_name1 (str): Name of the first parameter (rows in the heatmap).
        param_name2 (str): Name of the second parameter (columns in the heatmap).
        score_column (str): Column in the CSV that contains the scores to be plotted.
    """
    # Load the CSV
    results_df = pd.read_csv(csv_path)

    # Validate parameter columns
    if param_name1 not in results_df.columns or param_name2 not in results_df.columns:
        raise ValueError(f"Columns {param_name1} and {param_name2} must exist in the CSV.")

    if score_column not in results_df.columns:
        raise ValueError(f"Score column {score_column} must exist in the CSV.")

    # Aggregate duplicates by taking the mean
    aggregated_df = results_df.groupby([param_name1, param_name2])[score_column].mean().reset_index()

    # Pivot the aggregated data for heatmap
    pivot_table = aggregated_df.pivot(index=param_name1, columns=param_name2, values=score_column)

    # Check for missing values in the pivot table
    if pivot_table.isnull().values.any():
        print("[WARNING] Missing values detected in the pivot table. They will appear as blank areas in the heatmap.")

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".4f",
        cmap="viridis",
        cbar_kws={'label': 'Mean Test Score'},
    )
    plt.title(f"Grid Search Scores: {param_name1} vs {param_name2}")
    plt.xlabel(param_name2)
    plt.ylabel(param_name1)
    plt.show()

# Example usage
csv_path = "grid_search_results.csv"
plot_grid_search_from_csv(csv_path, param_name1='param_colsample_bytree', param_name2='param_n_estimators')