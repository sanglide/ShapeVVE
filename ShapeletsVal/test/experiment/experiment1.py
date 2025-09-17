import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare
import numpy as np

df = pd.read_csv("experiment.csv")
df.set_index('Dataset', inplace=True)

# Handle missing values (drop rows with missing values)
df = df.dropna()
df.drop(df.columns[-1], axis=1, inplace=True)

print("First few rows of data:")
print(df.head())
print("\nData shape:", df.shape)
print("\nMissing values per column:")
print(df.isnull().sum())

# Method 1: Friedman test using scipy.stats.friedmanchisquare
print("\n" + "="*60)
print("Friedman Test Results (using scipy.stats.friedmanchisquare):")
print("="*60)

# Prepare data: each algorithm as a separate column
data_arrays = [df[col].values for col in df.columns]

# Perform Friedman test
stat, p_value = friedmanchisquare(*data_arrays)

print(f"Friedman statistic (χ²): {stat:.4f}")
print(f"p-value: {p_value:.6f}")

if p_value < 0.05:
    print("Result: Reject null hypothesis at 0.05 significance level, significant differences exist in algorithm performance")
else:
    print("Result: Fail to reject null hypothesis at 0.05 significance level, no significant differences in algorithm performance")

algorithms = df.columns.tolist()
algo = 'ShapeVVE'

algo_scores = df[algo]
detailed_results = []

for other_algo in algorithms:
    if algo != other_algo:
        other_scores = df[other_algo]
        try:
            # Perform Wilcoxon signed-rank test
            stat, p_value = wilcoxon(algo_scores, other_scores, zero_method='pratt', method='approx')
            detailed_results.append({
                'Algorithm1': algo,
                'Algorithm2': other_algo,
                'Wilcoxon Statistic': stat,
                'p-value': p_value,
                'Significant (p < 0.05)': p_value < 0.05
            })
        except ValueError as e:
            print(f"Warning: Wilcoxon test failed for {algo} vs {other_algo}: {e}")
            detailed_results.append(np.nan)

# Create results DataFrame
result_df = pd.DataFrame(detailed_results)

print("\n" + "=" * 60)
print("Wilcoxon test results comparing ShapeVVE with all other algorithms:")
print("=" * 60)
print(result_df)
result_df.to_csv('result.csv', index=False)