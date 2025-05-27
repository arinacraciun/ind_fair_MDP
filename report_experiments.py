import os
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import yaml
import copy 

# Import custom classes from the other file
from one_run import (
    EpisodicFairnessEnv,
    FairDQN,
    FairnessEvalCallback,
    train_and_save_proxy_model,
    get_or_train_proxy_model
)

### 1. Helper Functions

def prepare_data(config):
    """Prepares data based on the loaded configuration."""
    cols = config['dataset_columns']
    knn_cfg = config['knn_settings']
    proxy_path = config['file_paths']['proxy_model']
    
    df = pd.read_csv(config['file_paths']['input_data'])
    
    # Binarize target
    target_proxy_col = f"{cols['target_original']}_binary"
    df[target_proxy_col] = (df[cols['target_original']] == cols['target_positive_class']).astype(int)

    # Define feature lists from config
    potential_features = [c for c in df.columns if c not in [cols['target_original'], target_proxy_col]]
    static_feats = [c for c in potential_features if c not in cols['driftable_features']]
    ordered_feats = static_feats + cols['driftable_features']
    
    states = df[ordered_feats].values.astype(np.float32)

    # kNN pairs from config
    nbrs = NearestNeighbors(n_neighbors=knn_cfg['k_neighbors']).fit(states)
    dists, idxs = nbrs.kneighbors(states)
    sims = np.exp(-(dists**2) / (knn_cfg['sigma']**2))
    fair_pairs = [(i, int(idxs[i, j]), float(sims[i, j]))
                  for i in range(states.shape[0])
                  for j in range(knn_cfg['k_neighbors'])]

    # Ensure proxy model exists
    proxy_model = get_or_train_proxy_model(
        model_path=proxy_path,
        df_train=df,
        static_feature_names_list=static_feats,
        target_column_name_proxy=target_proxy_col
    )

    return df, states, fair_pairs, static_feats, ordered_feats, proxy_model

def make_env_fn(config, df, static_feats, ordered_feats, proxy_model):
    """Creates an environment creation function from a given config."""
    cols = config['dataset_columns']
    
    def _fn():
        # Using the generic drift_logic from the config
        return EpisodicFairnessEnv(
            full_dataset_df=df.copy(),
            static_feature_names_list=static_feats,
            driftable_feature_names_list=cols['driftable_features'],
            ordered_feature_names_list=ordered_feats,
            target_column_name_for_proxy_training=f"{cols['target_original']}_binary",
            proxy_model=proxy_model,
            drift_logic=config['drift_logic']
        )
    return _fn

def plot_metrics(metrics_df, filename, config):
    """Plots all metrics and drifts."""
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

    # Drift Plots
    driftable_features = config['dataset_columns']['driftable_features']
    for i, feat in enumerate(driftable_features):
        plot_index = 9 + i
        
        ax = plt.subplot(2, 5, plot_index)
        safe_feat_name = feat.replace('.', '_')
        ax.plot(metrics_df["step"], metrics_df[f"drift_pct_{safe_feat_name}_grp0"], label="Group 0")
        ax.plot(metrics_df["step"], metrics_df[f"drift_pct_{safe_feat_name}_grp1"], label="Group 1")
        ax.set_xlabel("Timestep"); ax.set_ylabel("% Drift"); ax.legend()
        ax.set_title(f"Average % Drift: {feat}")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def update_config(config, key_path, value):
    """Updates a nested dictionary using a dot-separated key path."""
    keys = key_path.split('.')
    d = config
    for key in keys[:-1]:
        d = d[key]
    d[keys[-1]] = value
    return config


###  2. The Core Experiment Runner
def run_single_experiment(config, seed, data_artifacts, exp_tag):
    """Runs a single experiment for a given config and seed."""
    df, states, fair_pairs, static_feats, ordered_feats, proxy_model = data_artifacts
    h_params = config['fair_dqn_params']
    eval_cfg = config['evaluation']
    cols = config['dataset_columns']
    
    print(f"\n--- Running Experiment: {exp_tag} ---")

    # Set seeds for this specific run
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Create environment and model from config
    env_fn = make_env_fn(config, df, static_feats, ordered_feats, proxy_model)
    train_env = env_fn()
    train_env.reset(seed=seed)
    
    device = config['training']['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = FairDQN(
        "MlpPolicy", train_env,
        all_dataset_states=states,
        fair_pairs_list=fair_pairs,
        seed=seed,
        verbose=0,
        device=device,
        **h_params
    )

    callback = FairnessEvalCallback(
        full_df=df,
        initial_undrifted_states=states,
        fair_pairs_list=fair_pairs,
        true_labels_initial=df[f"{cols['target_original']}_binary"].values,
        sensitive_feature_name=cols['sensitive_attribute'],
        all_feature_names_ordered_list=ordered_feats,
        driftable_feature_names_list=cols['driftable_features'],
        eval_env_creator=env_fn,
        **eval_cfg
    )

    # Train the model
    model.learn(total_timesteps=config['training']['total_timesteps'], callback=callback)
    train_env.close()

    # Save results
    metrics_df = callback.get_metrics_log_df()
    metrics_df = metrics_df[metrics_df['step'] > model.learning_starts].reset_index(drop=True)

    csv_path = os.path.join(config['file_paths']['metrics_output_dir'], f"metrics_{exp_tag}.csv")
    metrics_df.to_csv(csv_path, index=False)

    plot_path = os.path.join(config['file_paths']['plots_output_dir'], f"plot_{exp_tag}.png")
    plot_metrics(metrics_df, plot_path, config)
    
    print(f"Finished {exp_tag}. Results saved.")


### 3. Main
if __name__ == '__main__':
    # Load the base configuration
    with open('config_experiments.yaml', 'r') as f:
        base_config = yaml.safe_load(f)

    # Prepare data once
    print("Preparing data, kNN pairs, and proxy model...")
    data_artifacts = prepare_data(base_config)
    
    # Create output directories from config
    os.makedirs(base_config['file_paths']['metrics_output_dir'], exist_ok=True)
    os.makedirs(base_config['file_paths']['plots_output_dir'], exist_ok=True)

    seeds = base_config['training']['seeds']

    ### FIRST RUN BASELINES
    print("\n\n--- Running Explicit Baseline (lambda=0, drift=0) ---")
    baseline_config = copy.deepcopy(base_config)
    
    # Manually set BOTH baseline conditions
    update_config(baseline_config, 'fair_dqn_params.lambda_fair', 0.0)
    
    # Find the 'no_drift' setting in the experiment sweeps to use it
    no_drift_setting = next((item for item in base_config['experiment_sweeps']['drift_logic'] if item['name'] == 'no_drift'), None)
    baseline_config['drift_logic'] = no_drift_setting['value']

    for seed in seeds:
        # Use a special tag for the baseline
        baseline_exp_tag = f"baseline_seed{seed}"
        run_single_experiment(baseline_config, seed, data_artifacts, baseline_exp_tag)
    
    print("\n--- Baseline Runs Finished. Starting Parameter Sweeps. ---")

    ### THEN RUN EXPERIMENTS
    # Get experiment definitions from config
    sweeps = base_config['experiment_sweeps']
    

    # Loop through each parameter sweep defined in the config
    for param_path, values in sweeps.items():
        for value_setting in values:
            # Create a deep copy of the base config for each run
            exp_config = copy.deepcopy(base_config)
            
            # The way we update the config depends on the parameter
            if param_path == 'drift_logic':
                # For complex objects like drift, the value is a dict with 'name' and 'value'
                exp_config['drift_logic'] = value_setting['value']
                param_value_str = value_setting['name']
            else:
                # For simple values, we update the nested dict
                update_config(exp_config, param_path, value_setting)
                param_value_str = str(value_setting)
            
            # Loop through all seeds for this configuration
            for seed in seeds:
                # Create a unique tag for this specific experiment run
                exp_tag = f"{param_path.split('.')[-1]}_{param_value_str}_seed{seed}"
                
                # Run the experiment with the modified config
                run_single_experiment(exp_config, seed, data_artifacts, exp_tag)
                
    print("\n\nAll experiments finished!")
