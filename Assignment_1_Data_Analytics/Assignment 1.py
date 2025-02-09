import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.stats import wilcoxon
from scipy.spatial.distance import mahalanobis

# Step 1: Simulate Patient Data
# Simulating patient data with baseline characteristics (e.g., pain, urgency, frequency)
# and assigning random treatment times to a subset of patients.
np.random.seed(42)
n_patients = 400

data = pd.DataFrame({
    'patient_id': np.arange(n_patients),
    'baseline_pain': np.random.randint(0, 10, n_patients),
    'baseline_urgency': np.random.randint(0, 10, n_patients),
    'baseline_frequency': np.random.randint(0, 10, n_patients),
    'treatment_time': np.random.choice([3, 6, 9, 12, None], size=n_patients, p=[0.2, 0.2, 0.2, 0.2, 0.2])
})
data['treated'] = data['treatment_time'].notna().astype(int)  # Flagging treated patients

# Step 2: Probability Distribution Analysis
# Visualizing a binomial distribution to understand probability mass function (PMF)
# Example parameters: n=10 (number of trials), p=0.5 (probability of success per trial)
def plot_binomial_distribution(n, p):
    from scipy.stats import binom
    r_values = np.arange(n + 1)
    pmf_values = binom.pmf(r_values, n, p)
    
    plt.figure(figsize=(8, 5))
    plt.bar(r_values, pmf_values, color='skyblue', edgecolor='black')
    plt.xlabel("Number of Successes (r)")
    plt.ylabel("Probability Mass Function (PMF)")
    plt.title(f"Binomial Distribution (n={n}, p={p})")
    plt.xticks(r_values)
    plt.show()

plot_binomial_distribution(10, 0.5)

# Step 3: Matching using Mahalanobis Distance
# Matching treated patients to control patients based on baseline characteristics
# using Mahalanobis Distance and Hungarian Algorithm.
def compute_mahalanobis_distance(treated_df, control_df):
    cov_matrix = np.cov(treated_df[['baseline_pain', 'baseline_urgency', 'baseline_frequency']].T)
    inv_cov_matrix = np.linalg.pinv(cov_matrix)  # Using pseudoinverse to avoid singular matrix issues
    distance_matrix = np.zeros((len(treated_df), len(control_df)))
    
    for i, (_, t_row) in enumerate(treated_df.iterrows()):
        for j, (_, c_row) in enumerate(control_df.iterrows()):
            diff = t_row[['baseline_pain', 'baseline_urgency', 'baseline_frequency']].values - \
                   c_row[['baseline_pain', 'baseline_urgency', 'baseline_frequency']].values
            distance_matrix[i, j] = mahalanobis(diff, np.zeros_like(diff), inv_cov_matrix)
    
    return distance_matrix

treated = data[data['treated'] == 1].reset_index(drop=True)
controls = data[data['treated'] == 0].reset_index(drop=True)

if len(treated) > 0 and len(controls) > 0:
    # Calculating distances and matching treated to control patients
    distance_matrix = compute_mahalanobis_distance(treated, controls)
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    matched_pairs = pd.DataFrame({
        'treated_id': treated.iloc[row_ind]['patient_id'].values,
        'control_id': controls.iloc[col_ind]['patient_id'].values,
        'distance': distance_matrix[row_ind, col_ind]
    })
else:
    matched_pairs = pd.DataFrame(columns=['treated_id', 'control_id', 'distance'])

# Step 4: Simulating Post-Treatment Data with Controlled Variability
# Generating post-treatment outcomes by adding variability to baseline characteristics
# and applying treatment effects (treated patients have different outcomes).
data['post_treatment_pain'] = data['baseline_pain'] - np.random.randint(0, 3, n_patients) - data['treated']
data['post_treatment_urgency'] = data['baseline_urgency'] - np.random.randint(0, 3, n_patients) - data['treated']
data['post_treatment_frequency'] = data['baseline_frequency'] - np.random.randint(0, 3, n_patients) - data['treated']

# Step 5: Statistical Analysis with Effect Size Calculation
# Comparing outcomes (treated vs. control) using Wilcoxon Signed-Rank Test.
# Effect size is calculated as the mean difference divided by the standard deviation.
if not matched_pairs.empty:
    treated_outcomes = data.set_index('patient_id').loc[matched_pairs['treated_id'], ['post_treatment_pain', 'post_treatment_urgency', 'post_treatment_frequency']]
    control_outcomes = data.set_index('patient_id').loc[matched_pairs['control_id'], ['post_treatment_pain', 'post_treatment_urgency', 'post_treatment_frequency']]
    
    for col in treated_outcomes.columns:
        stat, p = wilcoxon(treated_outcomes[col], control_outcomes[col])
        effect_size = (treated_outcomes[col].mean() - control_outcomes[col].mean()) / (treated_outcomes[col].std() + 1e-10)  # Avoid division by zero
        print(f'Wilcoxon test for {col}: statistic={stat}, p-value={p}, effect size={effect_size}')
else:
    print("No matched pairs available for statistical analysis.")

# Step 6: Improved Visualization
# Creating boxplots to compare post-treatment outcomes between treated and control groups.
plt.figure(figsize=(12, 5))
if not matched_pairs.empty:
    for i, col in enumerate(['post_treatment_pain', 'post_treatment_urgency', 'post_treatment_frequency']):
        plt.subplot(1, 3, i+1)
        plt.boxplot([treated_outcomes[col], control_outcomes[col]], labels=['Treated', 'Control'])
        plt.title(col.replace('_', ' ').title())
    plt.show()
else:
    print("No matched pairs available for visualization.")
