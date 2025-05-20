import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

### Baselines
# Load baseline data for each seed
df42 = pd.read_csv('metrics/metrics_baseline_seed42.csv')
df43 = pd.read_csv('metrics/metrics_baseline_seed43.csv')
df44 = pd.read_csv('metrics/metrics_baseline_seed44.csv')

# Merge dataframes
all_dfs = [df42, df43, df44]
merged_df = pd.concat(all_dfs)

# Group by step and calculate mean and std for each metric
grouped_df = merged_df.groupby("step").agg(
    {
        "mean_reward_vs_Yproxy": ["mean", "std"],
        "cfd_on_initial": ["mean", "std"],
        "consistency_on_initial": ["mean", "std"],
        "dp_gap_episodic": ["mean", "std"],
        "eo_tpr_gap_episodic": ["mean", "std"],
        "eo_fpr_gap_episodic": ["mean", "std"]
    }
).reset_index()

# Flatten multi-level columns for easier plotting
grouped_df.columns = ['step', 
                      'mean_reward_vs_Yproxy_mean', 'mean_reward_vs_Yproxy_std', 
                      'cfd_on_initial_mean', 'cfd_on_initial_std',
                      'consistency_on_initial_mean', 'consistency_on_initial_std',
                      'dp_gap_episodic_mean', 'dp_gap_episodic_std',
                      'eo_tpr_gap_episodic_mean', 'eo_tpr_gap_episodic_std',
                      'eo_fpr_gap_episodic_mean', 'eo_fpr_gap_episodic_std']

# Save summary stats to CSV for the report
summary_file = 'report_metrics/baseline_summary.csv'
grouped_df.to_csv(summary_file, index=False)

# Create a single figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# Plot 1: Mean Reward
axes[0, 0].plot(grouped_df["step"], grouped_df["mean_reward_vs_Yproxy_mean"], label="Mean Reward vs Y_proxy")
axes[0, 0].fill_between(grouped_df["step"], 
                         grouped_df["mean_reward_vs_Yproxy_mean"] - grouped_df["mean_reward_vs_Yproxy_std"], 
                         grouped_df["mean_reward_vs_Yproxy_mean"] + grouped_df["mean_reward_vs_Yproxy_std"], 
                         alpha=0.2)
axes[0, 0].set_title("Baseline - Mean Reward vs Y_proxy")
axes[0, 0].set_xlabel("Training Steps")
axes[0, 0].set_ylabel("Mean Reward")
axes[0, 0].grid(True)
axes[0, 0].legend()

# Plot 2: Counterfactual Fairness Disparity (CFD)
axes[0, 1].plot(grouped_df["step"], grouped_df["cfd_on_initial_mean"], label="CFD on Initial States")
axes[0, 1].fill_between(grouped_df["step"], 
                         grouped_df["cfd_on_initial_mean"] - grouped_df["cfd_on_initial_std"], 
                         grouped_df["cfd_on_initial_mean"] + grouped_df["cfd_on_initial_std"], 
                         alpha=0.2)
axes[0, 1].set_title("Baseline - CFD on Initial States")
axes[0, 1].set_xlabel("Training Steps")
axes[0, 1].set_ylabel("CFD")
axes[0, 1].grid(True)
axes[0, 1].legend()

# Plot 3: Consistency
axes[1, 0].plot(grouped_df["step"], grouped_df["consistency_on_initial_mean"], label="Consistency on Initial States")
axes[1, 0].fill_between(grouped_df["step"], 
                         grouped_df["consistency_on_initial_mean"] - grouped_df["consistency_on_initial_std"], 
                         grouped_df["consistency_on_initial_mean"] + grouped_df["consistency_on_initial_std"], 
                         alpha=0.2)
axes[1, 0].set_title("Baseline - Consistency on Initial States")
axes[1, 0].set_xlabel("Training Steps")
axes[1, 0].set_ylabel("Consistency")
axes[1, 0].grid(True)
axes[1, 0].legend()

# Plot 4: Group Fairness (EO and DP gaps)
axes[1, 1].plot(grouped_df["step"], grouped_df["dp_gap_episodic_mean"], label="DP Gap")
axes[1, 1].fill_between(grouped_df["step"], 
                         grouped_df["dp_gap_episodic_mean"] - grouped_df["dp_gap_episodic_std"], 
                         grouped_df["dp_gap_episodic_mean"] + grouped_df["dp_gap_episodic_std"], 
                         alpha=0.2)
axes[1, 1].plot(grouped_df["step"], grouped_df["eo_tpr_gap_episodic_mean"], label="EO TPR Gap")
axes[1, 1].fill_between(grouped_df["step"], 
                         grouped_df["eo_tpr_gap_episodic_mean"] - grouped_df["eo_tpr_gap_episodic_std"], 
                         grouped_df["eo_tpr_gap_episodic_mean"] + grouped_df["eo_tpr_gap_episodic_std"], 
                         alpha=0.2)
axes[1, 1].plot(grouped_df["step"], grouped_df["eo_fpr_gap_episodic_mean"], label="EO FPR Gap")
axes[1, 1].fill_between(grouped_df["step"], 
                         grouped_df["eo_fpr_gap_episodic_mean"] - grouped_df["eo_fpr_gap_episodic_std"], 
                         grouped_df["eo_fpr_gap_episodic_mean"] + grouped_df["eo_fpr_gap_episodic_std"], 
                         alpha=0.2)
axes[1, 1].set_title("Baseline - Group Fairness (DP and EO Gaps)")
axes[1, 1].set_xlabel("Training Steps")
axes[1, 1].set_ylabel("Gap")
axes[1, 1].grid(True)
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("report_plots/baseline_combined.png")
plt.show()

### Lambda sweeps

# Directory containing lambda sweep metrics
metrics_dir = 'metrics/'
lambda_values = [0.0, 1.0, 5.0]
seeds = [42, 43, 44]

# Prepare empty dataframe for combined results
all_metrics = []

# Load and aggregate data across all lambda values and seeds
for lambda_val in lambda_values:
    temp_dfs = []
    for seed in seeds:
        file_path = os.path.join(metrics_dir, f'metrics_lambda_{lambda_val}_seed{seed}.csv')
        df = pd.read_csv(file_path)
        df['lambda'] = lambda_val  # Add lambda as a column for grouping later
        temp_dfs.append(df)
    # Combine all seeds for this lambda
    combined_df = pd.concat(temp_dfs)
    all_metrics.append(combined_df)

# Merge all lambdas together
merged_df = pd.concat(all_metrics)

# Group by drift setting and step, then calculate mean and std for per-step analysis
grouped_df = merged_df.groupby(['lambda', 'step']).agg(
    {
        "mean_reward_vs_Yproxy": ["mean", "std"],
        "avg_lipschitz_penalty_train": ["mean", "std"],
        "lipschitz_on_final": ["mean", "std"],
        "cfd_on_initial": ["mean", "std"],
        "consistency_on_initial": ["mean", "std"],
        "dp_gap_episodic": ["mean", "std"],
        "eo_tpr_gap_episodic": ["mean", "std"],
        "eo_fpr_gap_episodic": ["mean", "std"],
        "drift_pct_capital_gain_grp0": ["mean", "std"],
        "drift_pct_capital_loss_grp0": ["mean", "std"],
        "drift_pct_capital_gain_grp1": ["mean", "std"],
        "drift_pct_capital_loss_grp1": ["mean", "std"]
    }
).reset_index()

# Flatten multi-level columns for easier plotting
grouped_df.columns = ['lambda', 'step', 
                      'mean_reward_vs_Yproxy_mean', 'mean_reward_vs_Yproxy_std',
                      "avg_lipschitz_penalty_train_mean", "avg_lipschitz_penalty_train_std", 
                      "lipschitz_on_final_mean", "lipschitz_on_final_std",
                      'cfd_on_initial_mean', 'cfd_on_initial_std',
                      'consistency_on_initial_mean', 'consistency_on_initial_std',
                      'dp_gap_episodic_mean', 'dp_gap_episodic_std',
                      'eo_tpr_gap_episodic_mean', 'eo_tpr_gap_episodic_std',
                      'eo_fpr_gap_episodic_mean', 'eo_fpr_gap_episodic_std',
                      'drift_pct_capital_gain_grp0_mean', 'drift_pct_capital_gain_grp0_std',
                      'drift_pct_capital_loss_grp0_mean', 'drift_pct_capital_loss_grp0_std',
                      'drift_pct_capital_gain_grp1_mean', 'drift_pct_capital_gain_grp1_std',
                      'drift_pct_capital_loss_grp1_mean', 'drift_pct_capital_loss_grp1_std']


# Calculate overall mean and std across all steps for final report
overall_summary_df = grouped_df.groupby('lambda').agg(
    {
        "mean_reward_vs_Yproxy_mean": ["mean", "std"],
        "avg_lipschitz_penalty_train_mean": ["mean", "std"],
        "lipschitz_on_final_mean": ["mean", "std"],
        "cfd_on_initial_mean": ["mean", "std"],
        "consistency_on_initial_mean": ["mean", "std"],
        "dp_gap_episodic_mean": ["mean", "std"],
        "eo_tpr_gap_episodic_mean": ["mean", "std"],
        "eo_fpr_gap_episodic_mean": ["mean", "std"],
        "drift_pct_capital_gain_grp0_mean": ["mean", "std"],
        "drift_pct_capital_loss_grp0_mean": ["mean", "std"],
        "drift_pct_capital_gain_grp1_mean": ["mean", "std"],
        "drift_pct_capital_loss_grp1_mean": ["mean", "std"]
    }
).reset_index()

# Flatten multi-level columns
overall_summary_df.columns = ['lambda',
                             'mean_reward_vs_Yproxy_mean', 'mean_reward_vs_Yproxy_std',
                             "avg_lipschitz_penalty_train_mean", "avg_lipschitz_penalty_train_std", 
                             "lipschitz_on_final_mean", "lipschitz_on_final_std",
                             'cfd_on_initial_mean', 'cfd_on_initial_std',
                             'consistency_on_initial_mean', 'consistency_on_initial_std',
                             'dp_gap_episodic_mean', 'dp_gap_episodic_std',
                             'eo_tpr_gap_episodic_mean', 'eo_tpr_gap_episodic_std',
                             'eo_fpr_gap_episodic_mean', 'eo_fpr_gap_episodic_std',
                             'drift_pct_capital_gain_grp0_mean', 'drift_pct_capital_gain_grp0_std',
                             'drift_pct_capital_loss_grp0_mean', 'drift_pct_capital_loss_grp0_std',
                             'drift_pct_capital_gain_grp1_mean', 'drift_pct_capital_gain_grp1_std',
                             'drift_pct_capital_loss_grp1_mean', 'drift_pct_capital_loss_grp1_std']

# Save overall summary stats to CSV for the final report
final_summary_file = 'report_metrics/lambda_sweep_summary_with_drift_overall.csv'
overall_summary_df.to_csv(final_summary_file, index=False)

# Create a single figure with 6 subplots (2x3 grid)
fig, axes = plt.subplots(2, 3, figsize=(24, 14))

# Plot 1: Mean Reward
for lambda_val in lambda_values:
    lambda_df = grouped_df[grouped_df['lambda'] == lambda_val]
    axes[0, 0].plot(lambda_df["step"], lambda_df["mean_reward_vs_Yproxy_mean"], label=f"λ = {lambda_val}")
    axes[0, 0].fill_between(lambda_df["step"], 
                             lambda_df["mean_reward_vs_Yproxy_mean"] - lambda_df["mean_reward_vs_Yproxy_std"], 
                             lambda_df["mean_reward_vs_Yproxy_mean"] + lambda_df["mean_reward_vs_Yproxy_std"], 
                             alpha=0.2)
axes[0, 0].set_title("Lambda Sweep - Mean Reward vs Y_proxy")
axes[0, 0].set_xlabel("Training Steps")
axes[0, 0].set_ylabel("Mean Reward")
axes[0, 0].grid(True)
axes[0, 0].legend()

# Plot 2: Counterfactual Fairness Disparity (CFD)
for lambda_val in lambda_values:
    lambda_df = grouped_df[grouped_df['lambda'] == lambda_val]
    axes[0, 1].plot(lambda_df["step"], lambda_df["cfd_on_initial_mean"], label=f"λ = {lambda_val}")
    axes[0, 1].fill_between(lambda_df["step"], 
                             lambda_df["cfd_on_initial_mean"] - lambda_df["cfd_on_initial_std"], 
                             lambda_df["cfd_on_initial_mean"] + lambda_df["cfd_on_initial_std"], 
                             alpha=0.2)
axes[0, 1].set_title("Lambda Sweep - CFD on Initial States")
axes[0, 1].set_xlabel("Training Steps")
axes[0, 1].set_ylabel("CFD")
axes[0, 1].grid(True)
axes[0, 1].legend()

# Plot 3: Consistency
for lambda_val in lambda_values:
    lambda_df = grouped_df[grouped_df['lambda'] == lambda_val]
    axes[0, 2].plot(lambda_df["step"], lambda_df["consistency_on_initial_mean"], label=f"λ = {lambda_val}")
    axes[0, 2].fill_between(lambda_df["step"], 
                             lambda_df["consistency_on_initial_mean"] - lambda_df["consistency_on_initial_std"], 
                             lambda_df["consistency_on_initial_mean"] + lambda_df["consistency_on_initial_std"], 
                             alpha=0.2)
axes[0, 2].set_title("Lambda Sweep - Consistency on Initial States")
axes[0, 2].set_xlabel("Training Steps")
axes[0, 2].set_ylabel("Consistency")
axes[0, 2].grid(True)
axes[0, 2].legend()

# Plot 3: Capital Gain Drift (Group 0)
for lambda_val in lambda_values:
    lambda_df = grouped_df[grouped_df['lambda'] == lambda_val]
    axes[1, 0].plot(lambda_df["step"], lambda_df["drift_pct_capital_gain_grp0_mean"], label=f"λ = {lambda_val}")
    axes[1, 0].fill_between(lambda_df["step"], 
                             lambda_df["drift_pct_capital_gain_grp0_mean"] - lambda_df["drift_pct_capital_gain_grp0_std"], 
                             lambda_df["drift_pct_capital_gain_grp0_mean"] + lambda_df["drift_pct_capital_gain_grp0_std"], 
                             alpha=0.2)
axes[1, 0].set_title("Lambda Sweep - Capital Gain Drifft (Group 0)")
axes[1, 0].set_xlabel("Training Steps")
axes[1, 0].set_ylabel("% Drift")
axes[1, 0].grid(True)
axes[1, 0].legend()

# Plot 3: Capital Gain Drift (Group 1)
for lambda_val in lambda_values:
    lambda_df = grouped_df[grouped_df['lambda'] == lambda_val]
    axes[1, 1].plot(lambda_df["step"], lambda_df["drift_pct_capital_gain_grp1_mean"], label=f"λ = {lambda_val}")
    axes[1, 1].fill_between(lambda_df["step"], 
                             lambda_df["drift_pct_capital_gain_grp1_mean"] - lambda_df["drift_pct_capital_gain_grp1_std"], 
                             lambda_df["drift_pct_capital_gain_grp1_mean"] + lambda_df["drift_pct_capital_gain_grp1_std"], 
                             alpha=0.2)
axes[1, 1].set_title("Lambda Sweep - Capital Gain Drifft (Group 1)")
axes[1, 1].set_xlabel("Training Steps")
axes[1, 1].set_ylabel("% Drift")
axes[1, 1].grid(True)
axes[1, 1].legend()

# Plot 4: Group Fairness (EO and DP gaps)
for lambda_val in lambda_values:
    lambda_df = grouped_df[grouped_df['lambda'] == lambda_val]
    axes[1, 2].plot(lambda_df["step"], lambda_df["dp_gap_episodic_mean"], label=f"DP Gap, λ = {lambda_val}")
    axes[1, 2].fill_between(lambda_df["step"], 
                             lambda_df["dp_gap_episodic_mean"] - lambda_df["dp_gap_episodic_std"], 
                             lambda_df["dp_gap_episodic_mean"] + lambda_df["dp_gap_episodic_std"], 
                             alpha=0.2)
axes[1, 2].set_title("Lambda Sweep - Group Fairness (DP Gap)")
axes[1, 2].set_xlabel("Training Steps")
axes[1, 2].set_ylabel("Gap")
axes[1, 2].grid(True)
axes[1, 2].legend()

plt.tight_layout()
plt.savefig("report_plots/lambda_sweep_combined_with_drift.png")
plt.show()


#### DRIFT

# Directory containing drift sweep metrics
metrics_dir = 'metrics/'
drift_settings = [(0.0, 0.0), (5.0, 5.0), (10.0, 10.0), (20.0, 20.0)]
seeds = [42, 43, 44]

# Prepare empty dataframe for combined results
all_metrics = []

# Load and aggregate data across all drift settings and seeds
for gain_pct, loss_pct in drift_settings:
    temp_dfs = []
    drift_name = f"{{'capital_gain_pct'_{gain_pct},'capital_loss_pct'_{loss_pct}}}"
    for seed in seeds:
        file_path = os.path.join(metrics_dir, f"metrics_drift_{{'capital_gain_pct'_{gain_pct},'capital_loss_pct'_{loss_pct}}}_seed{seed}.csv")
        df = pd.read_csv(file_path)
        df['drift_setting'] = drift_name  # Add drift setting as a column for grouping later
        temp_dfs.append(df)
    # Combine all seeds for this drift setting
    combined_df = pd.concat(temp_dfs)
    all_metrics.append(combined_df)

# Merge all drift settings together
merged_df = pd.concat(all_metrics)

# Group by drift setting and step, then calculate mean and std for per-step analysis
grouped_df = merged_df.groupby(['drift_setting', 'step']).agg(
    {
        "mean_reward_vs_Yproxy": ["mean", "std"],
        "avg_lipschitz_penalty_train": ["mean", "std"],
        "cfd_on_initial": ["mean", "std"],
        "consistency_on_initial": ["mean", "std"],
        "dp_gap_episodic": ["mean", "std"],
        "eo_tpr_gap_episodic": ["mean", "std"],
        "eo_fpr_gap_episodic": ["mean", "std"],
        "drift_pct_capital_gain_grp0": ["mean", "std"],
        "drift_pct_capital_loss_grp0": ["mean", "std"],
        "drift_pct_capital_gain_grp1": ["mean", "std"],
        "drift_pct_capital_loss_grp1": ["mean", "std"]
    }
).reset_index()

# Flatten multi-level columns for easier plotting
grouped_df.columns = ['drift_setting', 'step', 
                      'mean_reward_vs_Yproxy_mean', 'mean_reward_vs_Yproxy_std',
                      "avg_lipschitz_penalty_train_mean", "avg_lipschitz_penalty_train_std", 
                      'cfd_on_initial_mean', 'cfd_on_initial_std',
                      'consistency_on_initial_mean', 'consistency_on_initial_std',
                      'dp_gap_episodic_mean', 'dp_gap_episodic_std',
                      'eo_tpr_gap_episodic_mean', 'eo_tpr_gap_episodic_std',
                      'eo_fpr_gap_episodic_mean', 'eo_fpr_gap_episodic_std',
                      'drift_pct_capital_gain_grp0_mean', 'drift_pct_capital_gain_grp0_std',
                      'drift_pct_capital_loss_grp0_mean', 'drift_pct_capital_loss_grp0_std',
                      'drift_pct_capital_gain_grp1_mean', 'drift_pct_capital_gain_grp1_std',
                      'drift_pct_capital_loss_grp1_mean', 'drift_pct_capital_loss_grp1_std']


# Calculate overall mean and std across all steps for final report
overall_summary_df = grouped_df.groupby('drift_setting').agg(
    {
        "mean_reward_vs_Yproxy_mean": ["mean", "std"],
        "avg_lipschitz_penalty_train_mean": ["mean", "std"],
        "cfd_on_initial_mean": ["mean", "std"],
        "consistency_on_initial_mean": ["mean", "std"],
        "dp_gap_episodic_mean": ["mean", "std"],
        "eo_tpr_gap_episodic_mean": ["mean", "std"],
        "eo_fpr_gap_episodic_mean": ["mean", "std"],
        "drift_pct_capital_gain_grp0_mean": ["mean", "std"],
        "drift_pct_capital_loss_grp0_mean": ["mean", "std"],
        "drift_pct_capital_gain_grp1_mean": ["mean", "std"],
        "drift_pct_capital_loss_grp1_mean": ["mean", "std"]
    }
).reset_index()

# Flatten multi-level columns
overall_summary_df.columns = ['drift_setting',
                             'mean_reward_vs_Yproxy_mean', 'mean_reward_vs_Yproxy_std',
                             "avg_lipschitz_penalty_train_mean", "avg_lipschitz_penalty_train_std", 
                             'cfd_on_initial_mean', 'cfd_on_initial_std',
                             'consistency_on_initial_mean', 'consistency_on_initial_std',
                             'dp_gap_episodic_mean', 'dp_gap_episodic_std',
                             'eo_tpr_gap_episodic_mean', 'eo_tpr_gap_episodic_std',
                             'eo_fpr_gap_episodic_mean', 'eo_fpr_gap_episodic_std',
                             'drift_pct_capital_gain_grp0_mean', 'drift_pct_capital_gain_grp0_std',
                             'drift_pct_capital_loss_grp0_mean', 'drift_pct_capital_loss_grp0_std',
                             'drift_pct_capital_gain_grp1_mean', 'drift_pct_capital_gain_grp1_std',
                             'drift_pct_capital_loss_grp1_mean', 'drift_pct_capital_loss_grp1_std']

# Save overall summary stats to CSV for the final report
final_summary_file = 'report_metrics/drift_sweep_summary_overall.csv'
overall_summary_df.to_csv(final_summary_file, index=False)

# Plotting Drift Sweep Results
plt.figure(figsize=(24, 18))

# Plot 1: Mean Reward
plt.subplot(3, 2, 1)
for drift_name in grouped_df['drift_setting'].unique():
    df = grouped_df[grouped_df['drift_setting'] == drift_name]
    plt.plot(df['step'], df['mean_reward_vs_Yproxy_mean'], label=drift_name)
    plt.fill_between(df['step'], df['mean_reward_vs_Yproxy_mean'] - df['mean_reward_vs_Yproxy_std'],
                     df['mean_reward_vs_Yproxy_mean'] + df['mean_reward_vs_Yproxy_std'], alpha=0.2)
plt.title('Drift Sweep - Mean Reward vs Y_proxy')
plt.xlabel('Training Steps')
plt.ylabel('Mean Reward')
plt.legend()
plt.grid(True)

# Plot 2: CFD on Initial States
plt.subplot(3, 2, 2)
for drift_name in grouped_df['drift_setting'].unique():
    df = grouped_df[grouped_df['drift_setting'] == drift_name]
    plt.plot(df['step'], df['cfd_on_initial_mean'], label=drift_name)
    plt.fill_between(df['step'], df['cfd_on_initial_mean'] - df['cfd_on_initial_std'],
                     df['cfd_on_initial_mean'] + df['cfd_on_initial_std'], alpha=0.2)
plt.title('Drift Sweep - CFD on Initial States')
plt.xlabel('Training Steps')
plt.ylabel('CFD')
plt.legend()
plt.grid(True)

# Plot 2: consistency on Initial States
plt.subplot(3, 2, 3)
for drift_name in grouped_df['drift_setting'].unique():
    df = grouped_df[grouped_df['drift_setting'] == drift_name]
    plt.plot(df['step'], df['consistency_on_initial_mean'], label=drift_name)
    plt.fill_between(df['step'], df['consistency_on_initial_mean'] - df['consistency_on_initial_std'],
                     df['consistency_on_initial_mean'] + df['consistency_on_initial_std'], alpha=0.2)
plt.title('Drift Sweep - Consistency on Initial States')
plt.xlabel('Training Steps')
plt.ylabel('Consistency')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 4)
for drift_name in grouped_df['drift_setting'].unique():
    df = grouped_df[grouped_df['drift_setting'] == drift_name]
    plt.plot(df['step'], df['dp_gap_episodic_mean'], label=drift_name)
    plt.fill_between(df['step'], df['dp_gap_episodic_mean'] - df['dp_gap_episodic_std'],
                     df['dp_gap_episodic_mean'] + df['dp_gap_episodic_std'], alpha=0.2)
plt.title('Drift Sweep - DP Gap')
plt.xlabel('Training Steps')
plt.ylabel('Gap')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 5)
for drift_name in grouped_df['drift_setting'].unique():
    df = grouped_df[grouped_df['drift_setting'] == drift_name]
    plt.plot(df['step'], df['avg_lipschitz_penalty_train_mean'], label=drift_name)
    plt.fill_between(df['step'], df['avg_lipschitz_penalty_train_mean'] - df['avg_lipschitz_penalty_train_std'],
                     df['avg_lipschitz_penalty_train_mean'] + df['avg_lipschitz_penalty_train_std'], alpha=0.2)
plt.title('Drift Sweep - Average Lipschitz Penalty')
plt.xlabel('Training Steps')
plt.ylabel('Lipschitz Penalty')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("report_plots/drift_sweep_combined.png")
plt.show()

# Create a single figure with 2 subplots (1x2 grid)
fig, axes = plt.subplots(1, 2, figsize=(24, 14))
# Plot 3: Capital Gain Drift (Group 0)
for drift_name in grouped_df['drift_setting'].unique():
    df = grouped_df[grouped_df['drift_setting'] == drift_name]
    axes[0].plot(df["step"], df["drift_pct_capital_gain_grp0_mean"], label=drift_name)
    axes[0].fill_between(df["step"], 
                             df["drift_pct_capital_gain_grp0_mean"] - df["drift_pct_capital_gain_grp0_std"], 
                             df["drift_pct_capital_gain_grp0_mean"] + df["drift_pct_capital_gain_grp0_std"], 
                             alpha=0.2)
axes[0].set_title("Lambda Sweep - Capital Gain Drifft (Group 0)")
axes[0].set_xlabel("Training Steps")
axes[0].set_ylabel("% Drift")
axes[0].grid(True)
axes[0].legend()

# Plot 3: Capital Gain Drift (Group 1)
for drift_name in grouped_df['drift_setting'].unique():
    df = grouped_df[grouped_df['drift_setting'] == drift_name]
    axes[1].plot(df["step"], df["drift_pct_capital_gain_grp1_mean"], label=drift_name)
    axes[1].fill_between(df["step"], 
                             df["drift_pct_capital_gain_grp1_mean"] - df["drift_pct_capital_gain_grp1_std"], 
                             df["drift_pct_capital_gain_grp1_mean"] + df["drift_pct_capital_gain_grp1_std"], 
                             alpha=0.2)
axes[1].set_title("Lambda Sweep - Capital Gain Drifft (Group 1)")
axes[1].set_xlabel("Training Steps")
axes[1].set_ylabel("% Drift")
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.savefig("report_plots/drift_sweep_drift.png")
plt.show()

### PENALTY PARAMETER SWEEPS
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Directory containing fairness regularization parameter sensitivity metrics
metrics_dir = 'metrics/'
batch_sizes = [32, 128]
weighted_fracs = [0.0, 0.5, 1.0]
seeds = [42, 43, 44]

def load_metrics(file_pattern, batch_size=None, weighted_frac=None):
    temp_dfs = []
    for seed in seeds:
        file_path = os.path.join(metrics_dir, file_pattern.format(seed=seed))
        df = pd.read_csv(file_path)
        if batch_size is not None:
            df['batch_size'] = batch_size
            df['weighted_frac'] = 1.0  # Default for batch size sweeps
        if weighted_frac is not None:
            df['weighted_frac'] = weighted_frac
            df['batch_size'] = 128  # Default for weighted_frac sweeps
        temp_dfs.append(df)
    return pd.concat(temp_dfs)

# Load batch size sweep (default weighted_frac=1.0)
all_metrics = []
for batch_size in batch_sizes:
    batch_df = load_metrics(f'metrics_batch_size_pairs_{batch_size}_seed{{seed}}.csv', batch_size=batch_size)
    all_metrics.append(batch_df)

# Load weighted_frac sweep (default batch_size_pairs=128)
for weighted_frac in weighted_fracs:
    weighted_df = load_metrics(f'metrics_weighted_frac_{weighted_frac}_seed{{seed}}.csv', weighted_frac=weighted_frac)
    all_metrics.append(weighted_df)

# Combine all data
merged_df = pd.concat(all_metrics)

# Group by parameter settings and step, then calculate mean and std for per-step analysis
grouped_df = merged_df.groupby(['batch_size', 'weighted_frac', 'step']).agg(
    {
        "consistency_on_initial": ["mean", "std"],
        "cfd_on_initial": ["mean", "std"],
        "lipschitz_on_final": ["mean", "std"],
        "dp_gap_episodic": ["mean", "std"],
        "eo_tpr_gap_episodic": ["mean", "std"],
        "eo_fpr_gap_episodic": ["mean", "std"],
        "mean_reward_vs_Yproxy": ["mean", "std"]
    }
).reset_index()

# Flatten multi-level columns for easier plotting
grouped_df.columns = ['batch_size', 'weighted_frac', 'step',
                      'consistency_on_initial_mean', 'consistency_on_initial_std',
                      'cfd_on_initial_mean', 'cfd_on_initial_std',
                      'lipschitz_on_final_mean', 'lipschitz_on_final_std',
                      'dp_gap_episodic_mean', 'dp_gap_episodic_std',
                      'eo_tpr_gap_episodic_mean', 'eo_tpr_gap_episodic_std',
                      'eo_fpr_gap_episodic_mean', 'eo_fpr_gap_episodic_std',
                      'mean_reward_vs_Yproxy_mean', 'mean_reward_vs_Yproxy_std']

# Save summary stats to CSV
summary_file = 'report_metrics/fairness_reg_sensitivity_summary.csv'
grouped_df.to_csv(summary_file, index=False)

# Calculate overall mean and std across all steps for final report
overall_summary_df = grouped_df.groupby(['batch_size', 'weighted_frac']).agg(
    {
        "consistency_on_initial_mean": ["mean", "std"],
        "cfd_on_initial_mean": ["mean", "std"],
        "lipschitz_on_final_mean": ["mean", "std"],
        "dp_gap_episodic_mean": ["mean", "std"],
        "eo_tpr_gap_episodic_mean": ["mean", "std"],
        "eo_fpr_gap_episodic_mean": ["mean", "std"],
        "mean_reward_vs_Yproxy_mean": ["mean", "std"]
    }
).reset_index()

# Flatten multi-level columns
overall_summary_df.columns = ['batch_size', 'weighted_frac',
                             'consistency_on_initial_mean', 'consistency_on_initial_std',
                             'cfd_on_initial_mean', 'cfd_on_initial_std',
                             'lipschitz_on_final_mean', 'lipschitz_on_final_std',
                             'dp_gap_episodic_mean', 'dp_gap_episodic_std',
                             'eo_tpr_gap_episodic_mean', 'eo_tpr_gap_episodic_std',
                             'eo_fpr_gap_episodic_mean', 'eo_fpr_gap_episodic_std',
                             'mean_reward_vs_Yproxy_mean', 'mean_reward_vs_Yproxy_std']

# Save overall summary stats to CSV for the final report
final_summary_file = 'report_metrics/fairness_reg_sensitivity_summary_overall.csv'
overall_summary_df.to_csv(final_summary_file, index=False)

# Plot metrics with line plots
metrics_to_plot = ['consistency_on_initial_mean', 'cfd_on_initial_mean', 'lipschitz_on_final_mean']
plt.figure(figsize=(18, 16))

for i, metric in enumerate(metrics_to_plot, 1):
    plt.subplot(3, 1, i)
    valid_combinations = [(32, 1.0), (128, 0.0), (128, 0.5), (128, 1.0)]
    
    # Corrected indentation
    for batch_size, weighted_frac in valid_combinations:
        sub_df = grouped_df[(grouped_df['batch_size'] == batch_size) & (grouped_df['weighted_frac'] == weighted_frac)]
        plt.plot(sub_df['step'], sub_df[metric], label=f"Batch {batch_size}, WF {weighted_frac}")
        plt.fill_between(sub_df['step'], 
                         sub_df[metric] - sub_df[metric.replace('_mean', '_std')],
                         sub_df[metric] + sub_df[metric.replace('_mean', '_std')],
                         alpha=0.2)
    
    plt.title(metric.replace('_mean', '').replace('_', ' ').title())
    plt.xlabel("Training Steps")
    plt.ylabel(metric.split('_')[0].title())
    plt.legend()

plt.tight_layout()
plt.savefig("report_plots/fairness_reg_sensitivity_lineplots.png")
plt.show()