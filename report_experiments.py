import os
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from stable_baselines3.dqn.dqn import DQN as BaseDQN
from stable_baselines3 import DQN
from gymnasium import spaces
# Import your custom classes/functions
from final import (
    EpisodicFairnessEnv,
    FairDQN,
    FairnessEvalCallback,
    train_and_save_proxy_model,
    load_proxy_model
)

# ---------------------------
# 1. Customizable parameters
# ---------------------------
# Experiment values
LAMBDA_VALUES = [0.0, 1.0, 5.0]
BATCH_SIZE_PAIRS_VALUES = [32, 128]
WEIGHTED_FRAC_VALUES = [0.0, 0.5, 1.0]
DRIFT_SETTINGS_LIST = [
    {'capital_gain_pct': 0.0, 'capital_loss_pct': 0.0},  # No drift
    {'capital_gain_pct': 5.0, 'capital_loss_pct': 5.0},  # Low
    {'capital_gain_pct': 10.0, 'capital_loss_pct': 10.0},# Medium
    {'capital_gain_pct': 20.0, 'capital_loss_pct': 20.0} # High
]
# Common parameters
SEEDS = [42, 43, 44]
TOTAL_TIMESTEPS = 25_000
EVAL_FREQ = 1000
N_EVAL_REWARD = 50
N_EVAL_GROUP = 100

# Data & environment settings
DATA_PATH = "cleaned_adult.csv"
TARGET_COLUMN_ORIGINAL = 'income'
TARGET_COLUMN_PROXY = 'income_binary_gt_50k'
DRIFTABLE_FEATURES = ['capital.gain', 'capital.loss']

# Output dirs
METRICS_DIR = "metrics"
PLOTS_DIR = "plots"
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# -----------------------------------
# 2. Helper functions for setup
# -----------------------------------
def prepare_data():
    df = pd.read_csv(DATA_PATH)
    df[TARGET_COLUMN_PROXY] = (df[TARGET_COLUMN_ORIGINAL] == '>50K').astype(int)

    all_cols = df.columns.tolist()
    potential = [c for c in all_cols if c not in [TARGET_COLUMN_ORIGINAL, TARGET_COLUMN_PROXY]]
    static_feats = [c for c in potential if c not in DRIFTABLE_FEATURES]
    ordered = static_feats + DRIFTABLE_FEATURES

    states = df[ordered].values.astype(np.float32)

    # kNN pairs
    k = 5; sigma = 1.0
    nbrs = NearestNeighbors(n_neighbors=k).fit(states)
    dists, idxs = nbrs.kneighbors(states)
    sims = np.exp(- (dists**2) / (sigma**2))
    fair_pairs = [(i, int(idxs[i, j]), float(sims[i, j]))
                  for i in range(states.shape[0])
                  for j in range(k)]

    # Ensure proxy model
    proxy_path = "proxy_logistic_model.joblib"
    try:
        proxy = load_proxy_model(proxy_path)
    except FileNotFoundError:
        proxy = train_and_save_proxy_model(df, static_feats, TARGET_COLUMN_PROXY, proxy_path)

    return df, states, fair_pairs, static_feats, ordered


def make_env_fn(df, static_feats, ordered, drift_cfg, proxy):
    def _fn():
        return EpisodicFairnessEnv(
            full_dataset_df=df.copy(),
            static_feature_names_list=static_feats,
            driftable_feature_names_list=DRIFTABLE_FEATURES,
            ordered_feature_names_list=ordered,
            target_column_name_for_proxy_training=TARGET_COLUMN_PROXY,
            proxy_model=proxy,
            max_episode_steps=10,
            drift_config=drift_cfg
        )
    return _fn


def plot_metrics(metrics_df, filename):
    plt.figure(figsize=(36, 12))

    ax = plt.subplot(2, 5, 1)
    ax.plot(metrics_df["step"], metrics_df["mean_reward_vs_Yproxy"], label="Mean Reward (vs Y_proxy)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.set_title("Mean Reward")

    ax = plt.subplot(2, 5, 2)
    ax.plot(metrics_df["step"], metrics_df["avg_lipschitz_penalty_train"], label="Avg Lipschitz Penalty (Initial States)")
    ax.plot(metrics_df["step"], metrics_df["lipschitz_on_final"], label="Avg Lipschitz Penalty (Drifted States)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Lipschitz Penalty")
    ax.legend()
    ax.set_title("Average Lipschitz Penalty (Initial vs Drifted States)")

    ax = plt.subplot(2, 5, 3)
    ax.plot(metrics_df["step"], metrics_df["cfd_on_initial"], label="CFD (Initial States)")
    ax.plot(metrics_df["step"], metrics_df["cfd_on_final"], label="CFD (Final States)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Fairness Metric Value")
    ax.legend()
    ax.set_title("CFD (Initial vs. Drifted States)")

    ax = plt.subplot(2, 5, 4)
    ax.plot(metrics_df["step"], metrics_df["consistency_on_initial"], label="Consistency (Initial States)")
    ax.plot(metrics_df["step"], metrics_df["consistency_on_final"], label="Consistency (Final States)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Fairness Metric Value")
    ax.legend()
    ax.set_title("Consistency (Initial vs. Drifted States)")

    ax = plt.subplot(2, 5, 5)
    ax.plot(metrics_df["step"], metrics_df["dp_gap_episodic"], label="DP Gap (Episodic vs Y_proxy)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("DP Gap")
    ax.legend()
    ax.set_title("Demographic Parity Gap (Episodic)")

    ax = plt.subplot(2, 5, 6)
    ax.plot(metrics_df["step"], metrics_df["eo_tpr_gap_episodic"], label="EO TPR Gap (Episodic vs Y_proxy)")
    ax.plot(metrics_df["step"], metrics_df["eo_fpr_gap_episodic"], label="EO FPR Gap (Episodic vs Y_proxy)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("EO Gap")
    ax.legend()
    ax.set_title("Equalized Odds Gaps (Episodic)")

    ax = plt.subplot(2, 5, 7)
    ax.plot(metrics_df["step"], metrics_df["tpr_group0_ep"], label="TPR Group 0 (Episodic)")
    ax.plot(metrics_df["step"], metrics_df["tpr_group1_ep"], label="TPR Group 1 (Episodic)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("TPR")
    ax.legend()
    ax.set_title("True Positive Rates by Group (Episodic)")

    ax = plt.subplot(2, 5, 8)
    ax.plot(metrics_df["step"], metrics_df["fpr_group0_ep"], label="FPR Group 0 (Episodic)")
    ax.plot(metrics_df["step"], metrics_df["fpr_group1_ep"], label="FPR Group 1 (Episodic)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("FPR")
    ax.legend()
    ax.set_title("False Positive Rates by Group (Episodic)")

    ax = plt.subplot(2, 5, 9)
    ax.plot(metrics_df["step"], metrics_df["drift_pct_capital_gain_grp0"], label="Group 0")
    ax.plot(metrics_df["step"], metrics_df["drift_pct_capital_gain_grp1"], label="Group 1")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("% Drift")
    ax.legend()
    ax.set_title("Average % Drift: capital.gain")

    ax = plt.subplot(2, 5, 10)
    ax.plot(metrics_df["step"], metrics_df["drift_pct_capital_loss_grp0"], label="Group 0")
    ax.plot(metrics_df["step"], metrics_df["drift_pct_capital_loss_grp1"], label="Group 1")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("% Drift")
    ax.legend()
    ax.set_title("Average % Drift: capital.loss")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# -----------------------------------
# 3. Core runner
# -----------------------------------
def run_experiment(param_name, values, df, states, fair_pairs, static_feats, ordered):
    proxy_path = "proxy_logistic_model.joblib"
    proxy = load_proxy_model(proxy_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for val in values:
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            # Build drift settings or hyperparams
            drift_cfg = (val if param_name == 'drift' else DRIFT_SETTINGS_LIST[1])
            lambda_fair = val if param_name == 'lambda' else 1.0
            batch_pairs = val if param_name == 'batch_size_pairs' else 128
            wfrac = val if param_name == 'weighted_frac' else 1

            exp_tag = f"{param_name}_{str(val).replace(' ','')}_seed{seed}"

            # Create env & model
            env_fn = make_env_fn(df, static_feats, ordered, drift_cfg, proxy)
            train_env = env_fn()
            train_env.reset(seed=seed)

            model = FairDQN(
                "MlpPolicy", train_env,
                all_dataset_states=states,
                fair_pairs_list=fair_pairs,
                lambda_fair=lambda_fair,
                batch_size_pairs=batch_pairs,
                weighted_frac=wfrac,
                learning_rate=1e-3,
                buffer_size=100_000,
                learning_starts=5_000,
                batch_size=256,
                train_freq=2,
                gradient_steps=1,
                target_update_interval=1_000,
                exploration_fraction=0.1,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                tau=0.005,
                gamma=0.99,
                seed=seed,
                verbose=0,
                device=device
            )

            # Callback
            callback = FairnessEvalCallback(
                initial_undrifted_states=states,
                fair_pairs_list=fair_pairs,
                true_labels_initial=df[TARGET_COLUMN_PROXY].values,
                sensitive_feature_name='sex',
                all_feature_names_ordered_list=ordered,
                driftable_feature_names_list=DRIFTABLE_FEATURES,
                eval_env_creator=env_fn,
                eval_freq=EVAL_FREQ,
                n_eval_episodes_reward=N_EVAL_REWARD,
                n_eval_episodes_group_fairness=N_EVAL_GROUP,
                verbose=0
            )

            # Train
            model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
            train_env.close()

            # Collect metrics
            metrics_df = callback.get_metrics_log_df()
            metrics_df = metrics_df[metrics_df['step'] > model.learning_starts].reset_index(drop=True)

            # Save
            csv_path = os.path.join(METRICS_DIR, f"metrics_{exp_tag}.csv")
            metrics_df.to_csv(csv_path, index=False)

            plot_path = os.path.join(PLOTS_DIR, f"plot_{exp_tag}.png")
            plot_metrics(metrics_df, plot_path)

            print(f"Finished {exp_tag}, metrics -> {csv_path}, plot -> {plot_path}")

# ---------------------------
# 4. Main
# ---------------------------
if __name__ == '__main__':
    df, states, fair_pairs, static_feats, ordered = prepare_data()

    print("\nRunning baseline (no fairness, no drift)...")
    proxy_path = "proxy_logistic_model.joblib"
    proxy = load_proxy_model(proxy_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # use the first drift setting (zero) and default hyper‐params
    baseline_drift = DRIFT_SETTINGS_LIST[0]  # {'capital_gain_pct':0.0,'capital_loss_pct':0.0}
    baseline_lambda = 1.0
    baseline_batch_pairs = 128
    baseline_wfrac = 1.0

    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        exp_tag = f"baseline_seed{seed}"
        env_fn = make_env_fn(df, static_feats, ordered, baseline_drift, proxy)
        train_env = env_fn()
        train_env.reset(seed=seed)

        model = FairDQN(
            "MlpPolicy", train_env,
            all_dataset_states=states,
            fair_pairs_list=fair_pairs,
            lambda_fair=baseline_lambda,
            batch_size_pairs=baseline_batch_pairs,
            weighted_frac=baseline_wfrac,
            learning_rate=1e-3,
            buffer_size=100_000,
            learning_starts=5_000,
            batch_size=256,
            train_freq=2,
            gradient_steps=1,
            target_update_interval=1_000,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            tau=0.005,
            gamma=0.99,
            seed=seed,
            verbose=0,
            device=device
        )

        callback = FairnessEvalCallback(
            initial_undrifted_states=states,
            fair_pairs_list=fair_pairs,
            true_labels_initial=df[TARGET_COLUMN_PROXY].values,
            sensitive_feature_name='sex',
            all_feature_names_ordered_list=ordered,
            driftable_feature_names_list=DRIFTABLE_FEATURES,
            eval_env_creator=env_fn,
            eval_freq=EVAL_FREQ,
            n_eval_episodes_reward=N_EVAL_REWARD,
            n_eval_episodes_group_fairness=N_EVAL_GROUP,
            verbose=0
        )

        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
        train_env.close()

        # save & plot exactly as before
        metrics_df = callback.get_metrics_log_df()
        metrics_df = metrics_df[metrics_df['step'] > model.learning_starts].reset_index(drop=True)

        csv_path = os.path.join(METRICS_DIR, f"metrics_{exp_tag}.csv")
        metrics_df.to_csv(csv_path, index=False)

        plot_path = os.path.join(PLOTS_DIR, f"plot_{exp_tag}.png")
        plot_metrics(metrics_df, plot_path)

        print(f"Finished {exp_tag}, metrics -> {csv_path}, plot -> {plot_path}")

    # Individual experiments
    run_experiment('lambda', LAMBDA_VALUES, df, states, fair_pairs, static_feats, ordered)
    run_experiment('batch_size_pairs', BATCH_SIZE_PAIRS_VALUES, df, states, fair_pairs, static_feats, ordered)
    run_experiment('weighted_frac', WEIGHTED_FRAC_VALUES, df, states, fair_pairs, static_feats, ordered)
    # For drift, pass special marker
    run_experiment('drift', DRIFT_SETTINGS_LIST, df, states, fair_pairs, static_feats, ordered)
