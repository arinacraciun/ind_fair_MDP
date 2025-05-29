import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.ticker import FormatStrFormatter

### Baselines
df42 = pd.read_csv('experiments/metrics/metrics_baseline_seed42.csv')
df43 = pd.read_csv('experiments/metrics/metrics_baseline_seed43.csv')
df44 = pd.read_csv('experiments/metrics/metrics_baseline_seed44.csv')
df42['seed'] = 42
df43['seed'] = 43
df44['seed'] = 44
df_all = pd.concat([df42, df43, df44], ignore_index=True)
metrics_to_summarize = [
    "perf_success_rate_overall",
    "perf_success_rate_group_0",
    "perf_success_rate_group_1",
    "indiv_fairness_cfd_initial",
    "indiv_fairness_consistency_initial"
]
grouped = df_all.groupby("step")[metrics_to_summarize].agg(['mean', 'std']).reset_index()
grouped.columns = ['step'] + [f"{metric}_{stat}" for metric in metrics_to_summarize for stat in ['mean', 'std']]
for metric in metrics_to_summarize:
    mean_col = f"{metric}_mean"
    grouped[mean_col] = grouped[mean_col].rolling(window=3, min_periods=1).mean()

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Plot 1: Success Rate (overall and by group)
for group in ['overall', 'group_0', 'group_1']:
    mean_col = f"perf_success_rate_{group}_mean"
    std_col = f"perf_success_rate_{group}_std"
    axes[0].plot(grouped["step"], grouped[mean_col], label=f"Group {group}")
    axes[0].fill_between(grouped["step"], 
                         grouped[mean_col] - grouped[std_col],
                         grouped[mean_col] + grouped[std_col], alpha=0.2)
axes[0].set_title("Success Rate (Baseline)")
axes[0].set_ylabel("Success Rate")
axes[0].set_xlabel("Training Timesteps")
axes[0].legend()
axes[0].set_ylim(-0.05, 1.05)

# Plot 2: CFD (initial)
mean_col = "indiv_fairness_cfd_initial_mean"
std_col = "indiv_fairness_cfd_initial_std"
axes[1].plot(grouped["step"], grouped[mean_col], label="CFD (Initial)")
axes[1].fill_between(grouped["step"], 
                     grouped[mean_col] - grouped[std_col],
                     grouped[mean_col] + grouped[std_col], alpha=0.2)
axes[1].set_title("Counterfactual Fairness (Initial)")
axes[1].set_ylabel("CFD")
axes[1].set_xlabel("Training Timesteps")
axes[1].legend()

# Plot 3: Consistency (initial)
mean_col = "indiv_fairness_consistency_initial_mean"
std_col = "indiv_fairness_consistency_initial_std"
axes[2].plot(grouped["step"], grouped[mean_col], label="Consistency (Initial)")
axes[2].fill_between(grouped["step"], 
                     grouped[mean_col] - grouped[std_col],
                     grouped[mean_col] + grouped[std_col], alpha=0.2)
axes[2].set_title("Consistency Fairness (Initial)")
axes[2].set_ylabel("Consistency")
axes[2].set_xlabel("Training Timesteps")
axes[2].legend()
axes[2].yaxis.set_major_formatter(FormatStrFormatter('%.6f'))

plt.tight_layout()
os.makedirs("report_plots", exist_ok=True)
plt.savefig("report_plots/baseline_combined.png")
plt.close()

final_stats = df_all[df_all['step'] == df_all['step'].max()]
summary = final_stats[metrics_to_summarize].agg(['mean', 'std']).transpose().reset_index()
summary.columns = ['metric', 'mean', 'std']
os.makedirs("report_metrics", exist_ok=True)
summary.to_csv('report_metrics/baseline_summary.csv', index=False)






### Lambda sweeps
sns.set_theme(style="whitegrid")

lambda_values = [1.0, 5.0, 10.0]
seeds = [42, 43, 44]

lambda_dfs = {}

for lam in lambda_values:
    df_list = []
    for seed in seeds:
        path = f"experiments/metrics/metrics_lambda_fair_{lam}_seed{seed}.csv"
        df = pd.read_csv(path)
        df['seed'] = seed
        df['lambda_fair'] = lam
        df_list.append(df)
    lambda_dfs[lam] = pd.concat(df_list)

colors = sns.color_palette("tab10", len(lambda_values))
fig, axes = plt.subplots(2, 3, figsize=(22, 12))

# PLOT 1: Overall Success Rate
for i, lam in enumerate(lambda_values):
    df = lambda_dfs[lam].groupby("step").agg({
        "perf_success_rate_overall": ["mean", "std"]
    }).rolling(window=3, min_periods=1).mean().reset_index()
    mean_col = "perf_success_rate_overall"
    axes[0, 0].plot(df["step"], df[(mean_col, "mean")], label=f"λ={lam}", color=colors[i])
    axes[0, 0].fill_between(df["step"],
                            df[(mean_col, "mean")] - df[(mean_col, "std")],
                            df[(mean_col, "mean")] + df[(mean_col, "std")],
                            alpha=0.2, color=colors[i])
axes[0, 0].set_title("Overall Success Rate")
axes[0, 0].set_xlabel("Training Timesteps")
axes[0, 0].set_ylabel("Success Rate")
axes[0, 0].legend()
axes[0, 0].set_ylim(-0.05, 1.05)

# PLOT 2: CFD Initial
for i, lam in enumerate(lambda_values):
    df = lambda_dfs[lam].groupby("step").agg({
        "indiv_fairness_cfd_initial": ["mean", "std"]
    }).rolling(window=3, min_periods=1).mean().reset_index()
    mean_col = "indiv_fairness_cfd_initial"
    axes[0, 1].plot(df["step"], df[(mean_col, "mean")], label=f"λ={lam}", color=colors[i])
    axes[0, 1].fill_between(df["step"],
                            df[(mean_col, "mean")] - df[(mean_col, "std")],
                            df[(mean_col, "mean")] + df[(mean_col, "std")],
                            alpha=0.2, color=colors[i])
axes[0, 1].set_title("CFD: Counterfactual Fairness (Initial)")
axes[0, 1].set_xlabel("Training Timesteps")
axes[0, 1].yaxis.set_major_formatter(FormatStrFormatter('%.6f'))
axes[0, 1].set_ylabel("CFD (lower is better)")
axes[0, 1].legend()

# PLOT 3: CFD Final
for i, lam in enumerate(lambda_values):
    df = lambda_dfs[lam].groupby("step").agg({
        "indiv_fairness_cfd_final": ["mean", "std"]
    }).rolling(window=3, min_periods=1).mean().reset_index()
    mean_col = "indiv_fairness_cfd_final"
    axes[0, 2].plot(df["step"], df[(mean_col, "mean")], label=f"λ={lam}", color=colors[i])
    axes[0, 2].fill_between(df["step"],
                            df[(mean_col, "mean")] - df[(mean_col, "std")],
                            df[(mean_col, "mean")] + df[(mean_col, "std")],
                            alpha=0.2, color=colors[i])
axes[0, 2].set_title("CFD: Counterfactual Fairness (Final)")
axes[0, 2].set_xlabel("Training Timesteps")
axes[0, 2].set_ylabel("CFD (lower is better)")
axes[0, 2].legend()

# PLOT 4: Consistency Initial
for i, lam in enumerate(lambda_values):
    df = lambda_dfs[lam].groupby("step").agg({
        "indiv_fairness_consistency_initial": ["mean", "std"]
    }).rolling(window=3, min_periods=1).mean().reset_index()
    mean_col = "indiv_fairness_consistency_initial"
    axes[1, 0].plot(df["step"], df[(mean_col, "mean")], label=f"λ={lam}", color=colors[i])
    axes[1, 0].fill_between(df["step"],
                            df[(mean_col, "mean")] - df[(mean_col, "std")],
                            df[(mean_col, "mean")] + df[(mean_col, "std")],
                            alpha=0.2, color=colors[i])
axes[1, 0].set_title("Consistency Fairness (Initial)")
axes[1, 0].set_xlabel("Training Timesteps")
axes[1, 0].set_ylabel("Consistency (higher is better)")
axes[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.6f}'))
axes[1, 0].legend()

# PLOT 5: Drift - Capital Gain (averaged across G0 and G1)
for i, lam in enumerate(lambda_values):
    df = lambda_dfs[lam].groupby("step").agg({
        "drift_pct_capital_gain_group_0": ["mean", "std"],
        "drift_pct_capital_gain_group_1": ["mean", "std"]
    })
    
    # Average across group 0 and group 1
    df[("drift_avg", "mean")] = (df[("drift_pct_capital_gain_group_0", "mean")] + df[("drift_pct_capital_gain_group_1", "mean")]) / 2
    df[("drift_avg", "std")] = (df[("drift_pct_capital_gain_group_0", "std")]**2 + df[("drift_pct_capital_gain_group_1", "std")]**2)**0.5 / 2

    df = df.rolling(window=3, min_periods=1).mean().reset_index()
    
    axes[1, 1].plot(df["step"], df[("drift_avg", "mean")], label=f"λ={lam}", color=colors[i])
    axes[1, 1].fill_between(df["step"],
                            df[("drift_avg", "mean")] - df[("drift_avg", "std")],
                            df[("drift_avg", "mean")] + df[("drift_avg", "std")],
                            color=colors[i], alpha=0.2)
axes[1, 1].set_title("Drift: Capital Gain %")
axes[1, 1].set_xlabel("Training Timesteps")
axes[1, 1].set_ylabel("% Change in Capital Gain")
axes[1, 1].legend()

# PLOT 6: Drift - Capital Loss (averaged across G0 and G1)
for i, lam in enumerate(lambda_values):
    df = lambda_dfs[lam].groupby("step").agg({
        "drift_pct_capital_loss_group_0": ["mean", "std"],
        "drift_pct_capital_loss_group_1": ["mean", "std"]
    })
    
    # Average across group 0 and group 1
    df[("drift_avg", "mean")] = (df[("drift_pct_capital_loss_group_0", "mean")] + df[("drift_pct_capital_loss_group_1", "mean")]) / 2
    df[("drift_avg", "std")] = (df[("drift_pct_capital_loss_group_0", "std")]**2 + df[("drift_pct_capital_loss_group_1", "std")]**2)**0.5 / 2

    # Smooth
    df = df.rolling(window=3, min_periods=1).mean().reset_index()
    
    axes[1, 2].plot(df["step"], df[("drift_avg", "mean")], label=f"λ={lam}", color=colors[i])
    axes[1, 2].fill_between(df["step"],
                            df[("drift_avg", "mean")] - df[("drift_avg", "std")],
                            df[("drift_avg", "mean")] + df[("drift_avg", "std")],
                            color=colors[i], alpha=0.2)
axes[1, 2].set_title("Drift: Capital Loss %")
axes[1, 2].set_xlabel("Training Timesteps")
axes[1, 2].set_ylabel("% Change in Capital Loss")
axes[1, 2].legend()


plt.tight_layout()
os.makedirs("report_plots", exist_ok=True)
plt.savefig("report_plots/lambda_sweep_combined.png")
plt.close()

summary_rows = []
metrics_to_summarize = [
    "perf_success_rate_overall",
    "indiv_fairness_cfd_initial",
    "indiv_fairness_cfd_final",
    "indiv_fairness_consistency_initial",
    "indiv_fairness_consistency_final",
    "drift_pct_capital_gain_group_0",
    "drift_pct_capital_gain_group_1",
    "drift_pct_capital_loss_group_0",
    "drift_pct_capital_loss_group_1"
]

for lam in lambda_values:
    df = lambda_dfs[lam]
    final_step = df['step'].max()
    final_df = df[df['step'] == final_step]
    summary = final_df[metrics_to_summarize].agg(['mean', 'std']).transpose().reset_index()
    summary.columns = ['metric', 'mean', 'std']
    summary['lambda_fair'] = lam
    summary_rows.append(summary)

summary_df = pd.concat(summary_rows, ignore_index=True)
summary_df = summary_df[['lambda_fair', 'metric', 'mean', 'std']]  # reorder columns

summary_df.to_csv("report_metrics/lambda_sweep_summary.csv", index=False)






#### DRIFT

drift_levels = ["low", "medium", "high"]
seeds = [43, 44, 45]
base_path = "experiments/metrics"
rolling_window = 3

metrics_to_plot = {
    "perf_success_rate_overall": "Success Rate (Overall)",
    "avg_lipschitz_penalty_train": "Lipschitz Penalty",
    "indiv_fairness_consistency_initial": "Consistency (Initial)",
    "indiv_fairness_consistency_final": "Consistency (Final)",
    "indiv_fairness_cfd_initial": "CFD (Initial)",
    "indiv_fairness_cfd_final": "CFD (Final)",
}

all_data = {}
for drift in drift_levels:
    dfs = []
    for seed in seeds:
        filename = f"metrics_drift_logic_{drift}_drift_seed{seed}.csv"
        filepath = os.path.join(base_path, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df["seed"] = seed
            dfs.append(df)
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        all_data[drift] = combined

summary_data = {}
for drift, df in all_data.items():
    grouped = df.groupby("step")
    agg_df = grouped[list(metrics_to_plot.keys())].agg(['mean', 'std'])
    agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
    agg_df = agg_df.reset_index()
    agg_df = agg_df.rolling(window=rolling_window, min_periods=1).mean()
    summary_data[drift] = agg_df

summary_rows = []

for drift, df in all_data.items():
    for metric, label in metrics_to_plot.items():
        values = df[metric]
        summary_rows.append({
            "drift_level": drift,
            "metric": metric,
            "label": label,
            "mean": values.mean(),
            "std": values.std()
        })

summary_df = pd.DataFrame(summary_rows)

summary_out_path = "report_metrics/drift_sweep_summary.csv"
summary_df.to_csv(summary_out_path, index=False)

# Plot
plt.figure(figsize=(18, 12))
for i, (metric, label) in enumerate(metrics_to_plot.items(), 1):
    plt.subplot(3, 2, i)
    for drift in drift_levels:
        df = summary_data[drift]
        steps = df["step"]
        mean = df[f"{metric}_mean"]
        std = df[f"{metric}_std"]
        plt.plot(steps, mean, label=f"{drift.title()} Drift")
        plt.fill_between(steps, mean - std, mean + std, alpha=0.2)
    
    plt.title(label)
    plt.xlabel("Training Timesteps")
    plt.ylabel(label)
    plt.grid(True, alpha=0.3)
    plt.legend()

plt.tight_layout()
plot_path = "report_plots/plot_drift_sweeps.png"
plt.savefig(plot_path)
print(f"Saved final plot at: {plot_path}")
plt.show()





### PENALTY PARAMETER SWEEPS
experiments = {
    "Batch 32 WF 1.0": "metrics_batch_size_pairs_32_seed42.csv",
    "Batch 128 WF 0.0": "metrics_weighted_frac_0.0_seed42.csv",
    "Batch 128 WF 0.5": "metrics_weighted_frac_0.5_seed42.csv",
    "Batch 128 WF 1.0": "metrics_weighted_frac_1.0_seed42.csv",
}

base_path = "experiments/metrics"

metrics_to_plot = {
    "perf_success_rate_overall": "Success Rate (Overall)",
    "avg_lipschitz_penalty_train": "Lipschitz Penalty",
    "indiv_fairness_consistency_initial": "Consistency (Initial)",
    "indiv_fairness_cfd_initial": "CFD (Initial)",
    "indiv_fairness_cfd_final": "CFD (Final)",
    "drift_pct_capital_gain_group_0": "Capital Gain Drift (Avg)",
    "drift_pct_capital_gain_group_1": "Capital Gain Drift (Avg)",
}

summary_rows = []
all_data = {}

for label, filename in experiments.items():
    filepath = os.path.join(base_path, filename)
    df = pd.read_csv(filepath)
    all_data[label] = df

    for metric, display in metrics_to_plot.items():
        if "Capital Gain Drift" in display:
            continue  
        values = df[metric]
        summary_rows.append({
            "experiment": label,
            "metric": metric,
            "label": display,
            "mean": values.mean(),
            "std": values.std()
        })

    gain_0 = df["drift_pct_capital_gain_group_0"]
    gain_1 = df["drift_pct_capital_gain_group_1"]
    gain_avg = (gain_0 + gain_1) / 2
    summary_rows.append({
        "experiment": label,
        "metric": "capital_gain_drift_avg",
        "label": "Capital Gain Drift (Avg)",
        "mean": gain_avg.mean(),
        "std": gain_avg.std()
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("report_metrics/param_sweep_summary.csv", index=False)

plot_metrics = [
    "perf_success_rate_overall",
    "avg_lipschitz_penalty_train",
    "indiv_fairness_consistency_initial",
    "indiv_fairness_cfd_initial",
    "indiv_fairness_cfd_final",
    "capital_gain_drift_avg"
]

label_map = {row["metric"]: row["label"] for row in summary_rows}

plt.figure(figsize=(18, 12))
for i, metric in enumerate(plot_metrics, 1):
    plt.subplot(3, 2, i)
    for label, df in all_data.items():
        if metric == "capital_gain_drift_avg":
            y = ((df["drift_pct_capital_gain_group_0"] + df["drift_pct_capital_gain_group_1"]) / 2).rolling(window=3, min_periods=1).mean()
            std = ((df["drift_pct_capital_gain_group_0"] + df["drift_pct_capital_gain_group_1"]) / 2).rolling(window=3, min_periods=1).std()
        else:
            y = df[metric].rolling(window=3, min_periods=1).mean()
            std = df[metric].rolling(window=3, min_periods=1).std()
        steps = df["step"]
        plt.plot(steps, y, label=label)
        plt.fill_between(steps, y - std, y + std, alpha=0.2)

    plt.title(label_map[metric])
    plt.xlabel("Training Timesteps")
    plt.ylabel(label_map[metric])
    plt.grid(True, alpha=0.3)
    plt.legend()

plt.tight_layout()
plt.savefig("report_plots/plot_sweep_params.png")
plt.show()
