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
)

### 1. Helper Functions

def prepare_data(config):
    cols = config['dataset_columns']
    knn_cfg = config['knn_settings']
    
    df = pd.read_csv(config['file_paths']['input_data'])

    ground_truth_target_col = f"{cols['target_original']}_binary"
    df[ground_truth_target_col] = (df[cols['target_original']] == cols['target_positive_class']).astype(int)

    # Define feature lists from config
    potential_features = [c for c in df.columns if c not in [cols['target_original'], ground_truth_target_col]]
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


    return df, states, fair_pairs, static_feats, ordered_feats

def make_env_fn(config, df, static_feats, ordered_feats, proxy_model): 
    ### Creates an environment creation function from a given config.
    cols = config['dataset_columns']
    reward_cfg = config['rewards']
    train_cfg = config['training']
    
    def _fn():
        return EpisodicFairnessEnv(
            full_dataset_df=df.copy(),
            static_feature_names_list=static_feats,
            driftable_feature_names_list=cols['driftable_features'],
            ordered_feature_names_list=ordered_feats,
            ground_truth_target_column=f"{cols['target_original']}_binary",
            drift_logic=config['drift_logic'],
            max_episode_steps=train_cfg['max_episode_steps'],
            terminal_reward=reward_cfg['terminal_reward'],
            terminal_penalty=reward_cfg['terminal_penalty'],
            step_cost=reward_cfg['step_cost']
        )
    return _fn

def plot_results(metrics_df, filename, config):
    """ ## CHANGED: Completely new plotting function for Path B metrics."""
    print(f"Generating plot dashboard and saving to {filename}...")
    plt.figure(figsize=(18, 10))
    sensitive_groups = [0, 1] # Assuming binary groups
    
    # Plot 1: Performance & Outcome Fairness (Success Rate)
    ax = plt.subplot(2, 3, 1)
    for group in sensitive_groups:
        ax.plot(metrics_df["step"], metrics_df[f"perf_success_rate_group_{group}"], label=f"Success Rate (Group {group})")
    ax.plot(metrics_df["step"], metrics_df["perf_success_rate_overall"], label="Overall", linestyle='--', color='k', alpha=0.7)
    ax.set_title("Agent Performance: Per-Group Success Rate")
    ax.set_xlabel("Training Timesteps"); ax.set_ylabel("Success Rate"); ax.legend(); ax.grid(True, alpha=0.4)
    ax.set_ylim(-0.05, 1.05)

    # Plot 2: Individual Fairness (Consistency)
    ax = plt.subplot(2, 3, 2)
    ax.plot(metrics_df["step"], metrics_df["indiv_fairness_consistency_initial"], label="Initial States")
    ax.plot(metrics_df["step"], metrics_df["indiv_fairness_consistency_final"], label="Final States")
    ax.set_title("Individual Fairness: Consistency Score")
    ax.set_xlabel("Training Timesteps"); ax.set_ylabel("Consistency (1 is best)"); ax.legend(); ax.grid(True, alpha=0.4)
    ax.set_ylim(-0.05, 1.05)
    
    # Plot 3: Individual Fairness (CFD)
    ax = plt.subplot(2, 3, 3)
    ax.plot(metrics_df["step"], metrics_df["indiv_fairness_cfd_initial"], label="Initial States")
    ax.plot(metrics_df["step"], metrics_df["indiv_fairness_cfd_final"], label="Final States")
    ax.set_title("Individual Fairness: Counterfactual Fairness")
    ax.set_xlabel("Training Timesteps"); ax.set_ylabel("CFD (0 is best)"); ax.legend(); ax.grid(True, alpha=0.4)
    
    # Plot 4: Training Fairness Penalty
    ax = plt.subplot(2, 3, 4)
    ax.plot(metrics_df["step"], metrics_df["avg_lipschitz_penalty_train"], color='purple')
    ax.set_title("Training Fairness Penalty (λ_fair)")
    ax.set_xlabel("Training Timesteps"); ax.set_ylabel("Penalty Value"); ax.grid(True, alpha=0.4)

    # Plots 5 & 6: Feature Drift
    driftable_features = config['dataset_columns']['driftable_features']
    for i, feat in enumerate(driftable_features):
        if i >= 2: break
        ax = plt.subplot(2, 3, 5 + i)
        safe_feat_name = feat.replace('.', '_')
        for group in sensitive_groups:
             ax.plot(metrics_df["step"], metrics_df[f"drift_pct_{safe_feat_name}_group_{group}"], label=f"Group {group}")
        ax.set_title(f"Feature Drift: {feat}")
        ax.set_xlabel("Training Timesteps"); ax.set_ylabel("Average % Change"); ax.legend(); ax.grid(True, alpha=0.4)

    plt.tight_layout(pad=3.0)
    plt.savefig(filename)
    plt.close()

def update_config(config, key_path, value):
    keys = key_path.split('.')
    d = config
    for key in keys[:-1]:
        d = d[key]
    d[keys[-1]] = value
    return config


###  2. The Core Experiment Runner
def run_single_experiment(config, seed, data_artifacts, exp_tag):
    ### Runs a single experiment for a given config and seed.
    df, states, fair_pairs, static_feats, ordered_feats = data_artifacts
    h_params = config['fair_dqn_params']
    eval_cfg = config['evaluation']
    cols = config['dataset_columns']
    
    print(f"\n--- Running Experiment: {exp_tag} ---")

    # Set seeds for this specific run
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Create environment and model from config
    env_fn = make_env_fn(config, df, static_feats, ordered_feats) ## CHANGED: No proxy model passed
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
        sensitive_feature_name=cols['sensitive_attribute'],
        all_feature_names_ordered_list=ordered_feats,
        driftable_feature_names_list=cols['driftable_features'],
        eval_env_creator=env_fn,
        sigma_final=config['knn_settings']['sigma'],
        k_final=config['knn_settings']['k_neighbors'],
        verbose=0,
        **eval_cfg
    )

    # Train the model
    model.learn(total_timesteps=config['training']['total_timesteps'], callback=callback)
    train_env.close()

    metrics_df = callback.get_metrics_log_df()
    if not metrics_df.empty:
        metrics_df = metrics_df[metrics_df['step'] > model.learning_starts].reset_index(drop=True)
        csv_path = os.path.join(config['file_paths']['metrics_output_dir'], f"metrics_{exp_tag}.csv")
        metrics_df.to_csv(csv_path, index=False)
        plot_path = os.path.join(config['file_paths']['plots_output_dir'], f"plot_{exp_tag}.png")
        plot_results(metrics_df, plot_path, config)
        print(f"Finished {exp_tag}. Results saved.")
    else:
        print(f"Finished {exp_tag}, but no metrics were logged.")


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
    update_config(baseline_config, 'fair_dqn_params.lambda_fair', 0.0)
    no_drift_setting = next((item for item in base_config['experiment_sweeps']['drift_logic'] if item['name'] == 'no_drift'), None)
    if no_drift_setting:
        baseline_config['drift_logic'] = no_drift_setting['value']
        for seed in seeds:
            run_single_experiment(baseline_config, seed, data_artifacts, f"baseline_seed{seed}")
    else:
        print("Warning: Could not find 'no_drift' configuration to run baseline.")

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
                if value_setting['name'] == 'no_drift': continue # Skip baseline condition, already ran it
                exp_config['drift_logic'] = value_setting['value']
                param_value_str = value_setting['name']
            else:
                update_config(exp_config, param_path, value_setting)
                param_value_str = str(value_setting)
            
            for seed in seeds:
                exp_tag = f"{param_path.split('.')[-1]}_{param_value_str}_seed{seed}"
                run_single_experiment(exp_config, seed, data_artifacts, exp_tag)
                
    print("\n\nAll experiments finished!")

