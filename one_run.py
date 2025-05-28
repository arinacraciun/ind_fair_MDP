# Import all the necessary libraries. We're bringing in tools for building the reinforcement
# learning environment (gymnasium), handling data (numpy, pandas), finding similar
# neighbors (sklearn), building and managing the RL agent (stable_baselines3, torch),
# and a few other utilities for plotting, saving models, and suppressing warnings.
import gymnasium as gym
import numpy as np
import pandas as pd
import random
import torch
import torch.nn.functional as F
from gymnasium import spaces
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix
from stable_baselines3.dqn.dqn import DQN as BaseDQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
from stable_baselines3.common.evaluation import evaluate_policy
from sklearn.linear_model import LogisticRegression
import joblib
import yaml
import os

import warnings
# Ignore some common warnings from these libraries to keep the output clean.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# --- Episodic Fairness Environment ---
# This is our custom reinforcement learning environment. An "episode" in this
# environment consists of the agent making a series of decisions for a single
# individual sampled from the dataset.
class EpisodicFairnessEnv(gym.Env):
    metadata = {'render_modes': [], 'render_fps': 4}

    def __init__(self,
                 full_dataset_df: pd.DataFrame,
                 static_feature_names_list: list[str],
                 driftable_feature_names_list: list[str],
                 ordered_feature_names_list: list[str], # To keep the observation data consistent.
                 ground_truth_target_column: str,
                 terminal_reward: float = 100.0,        ## Reward for a successful outcome
                 terminal_penalty: float = -100.0,       ## Penalty for an unsuccessful outcome
                 step_cost: float = 1.0,               ## Cost for taking an action
                 max_episode_steps: int = 10, # How many decisions to make per individual.
                 drift_logic: dict = None): # Settings for how features change.
        super().__init__()

        # --- Basic Environment Setup ---
        self.full_dataset = full_dataset_df.reset_index(drop=True)
        self.static_feature_names = static_feature_names_list
        self.driftable_feature_names = driftable_feature_names_list
        self.all_feature_names_ordered = ordered_feature_names_list
        self.ground_truth_target_column = ground_truth_target_column
        self.max_episode_steps = max_episode_steps

        # --- Reward and Drift Configuration ---
        self.drift_logic = drift_logic if drift_logic is not None else {}
        self.terminal_reward = terminal_reward
        self.terminal_penalty = terminal_penalty
        self.step_cost = step_cost

        # --- Define Action and Observation Spaces ---
        # The agent can take one of two actions (0 or 1).
        self.action_space = spaces.Discrete(2)
        # The observation is a vector of all the individual's features.
        num_features = len(self.all_feature_names_ordered)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32)

        # --- Initialize Episode-Specific Variables ---
        # These will be reset at the start of each new episode.
        self.current_row_index = -1
        self.current_static_features_df = None   # The unchangeable features for the current person.
        self.current_driftable_features_df = None # The changeable features for the current person.
        self.current_original_target_from_df = None # The actual original outcome from the dataset.
        self.episode_step_count = 0
        self.np_random = None # The random number generator for reproducibility.

    def _get_observation(self) -> np.ndarray:
        #Construct the observation vector.
        # Combine the feature sets into a single DataFrame.
        obs_data = pd.concat([
            self.current_static_features_df[self.static_feature_names],
            self.current_driftable_features_df[self.driftable_feature_names]
        ], axis=1)
        # Ensure the final order is correct and return as a numpy array.
        return obs_data[self.all_feature_names_ordered].values.flatten().astype(np.float32)

    def _apply_drift(self, action: int):
        #Modifies the driftable features based on the agent's action. This simulates
        #the real-world consequence of a decision.
        # Check if there's a drift rule for the action taken.
        if action not in self.drift_logic:
            return # No drift for this action

        # Get the rule for this action from our config dictionary.
        rule = self.drift_logic[action]
        feature_to_drift = rule['feature_to_drift']
        value = rule['value']

        # Get the current value of the feature for the individual in the episode.
        i = self.current_row_index
        current_val = self.current_driftable_features_df.at[i, feature_to_drift]
        
        new_val = current_val # Default to no change

        # Apply the drift by percentage.
        new_val = current_val * (1.0 + (value / 100.0))

        # Update the feature in the DataFrame for this episode.
        self.current_driftable_features_df.at[i, feature_to_drift] = new_val

    def reset(self, seed=None, options=None):
        # Starts a new episode. It picks a random person from the dataset and
        # initializes the state.

        super().reset(seed=seed) # This is important for seeding the random number generator.

        # Randomly sample one individual from the full dataset for this episode.
        self.current_row_index = self.np_random.integers(0, len(self.full_dataset))
        # Make a copy of this individual's data so we can modify it without affecting the original dataset.
        current_individual_data = self.full_dataset.iloc[[self.current_row_index]].copy()

        # Separate the static and driftable features for this individual.
        self.current_static_features_df = current_individual_data[self.static_feature_names]
        self.current_driftable_features_df = current_individual_data

        # Store the original target label from the dataset for reward and analysis.
        self.current_original_target_from_df = current_individual_data[self.ground_truth_target_column].iloc[0]
        self.episode_step_count = 0

        # Get the initial observation and create an info dictionary with helpful metadata.
        observation = self._get_observation()
        info = {
            "original_target": self.current_original_target_from_df,
            "sampled_row_index": self.current_row_index
        }
        return observation, info

    def step(self, action: int):
        ### Executes one time step within the episode.
        # Apply drift based on the action
        self._apply_drift(action)
        self.episode_step_count += 1

        # Check if the episode has ended
        terminated = self.episode_step_count >= self.max_episode_steps
        
        # Calculate the reward based on the new logic
        if terminated:
            # At the final step, give a large reward or penalty based on the true outcome
            if self.current_original_target_from_df == 1:
                reward = self.terminal_reward
            else:
                reward = self.terminal_penalty
        else:
            # For all other steps, apply the small intervention cost
            reward = -self.step_cost

        observation = self._get_observation()
        truncated = False # Not used
        info = {
            "original_target": self.current_original_target_from_df,
            "action_taken": action,
            "reward_received": reward,
            "is_terminal_step": terminated
        }
        return observation, reward, terminated, truncated, info

    # These are standard gym methods that we don't need for this setup, so we just pass.
    def close(self):
        pass

    def render(self):
        pass

# --- FairDQN (A DQN with a Fairness Penalty) ---
# This is our custom agent. It's based on the standard Deep Q-Network (DQN) from
# Stable Baselines3, but we've modified its training process to include a
# "fairness penalty".
class FairDQN(BaseDQN):
    """
    A modified DQN that adds a Lipschitz/consistency penalty to the loss function.
    This penalty encourages the model to make similar predictions for similar individuals.
    """
    def __init__(self,
                 policy,
                 env,
                 all_dataset_states: np.ndarray, # The initial, undrifted states of everyone.
                 fair_pairs_list: list, # A list of (person_i, person_j, similarity_score).
                 lambda_fair: float = 1.0, # How much to weigh the fairness penalty.
                 batch_size_pairs: int = 64, # How many pairs to check for fairness each step.
                 weighted_frac: float = 0.5, # The fraction of pairs to sample based on similarity.
                 **kwargs):
        super().__init__(policy, env, **kwargs)
        self.fair_pairs = fair_pairs_list
        self.lambda_fair = lambda_fair
        self.batch_size_pairs = batch_size_pairs
        self.weighted_frac = weighted_frac
        self.all_dataset_states = all_dataset_states # Used for calculating the fairness penalty.

        # Pre-calculate sampling probabilities for the pairs. More similar pairs will be sampled more often.
        self._sims = np.array([sim for (_, _, sim) in self.fair_pairs])
        self._probs = self._sims / (self._sims.sum() + 1e-8) # Add a small epsilon to avoid division by zero.

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """
        The main training loop. This is where we calculate both the standard
        DQN loss and our custom fairness penalty.
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        for _ in range(gradient_steps):
            # --- 1. Standard DQN Loss Calculation ---
            # Sample a batch of experiences (state, action, reward, next_state) from the replay buffer.
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # Calculate the target Q-values using the target network.
            with torch.no_grad():
                next_q_values = self.q_net_target(replay_data.next_observations)
                next_q_values, _ = next_q_values.max(dim=1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones.float()) * self.gamma * next_q_values

            # Get the current Q-values from the main network for the actions that were actually taken.
            current_q_values = self.q_net(replay_data.observations)
            current_q_values = torch.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # The standard DQN loss is the Mean Squared Error between current and target Q-values.
            loss_q = F.mse_loss(current_q_values.squeeze(1), target_q_values)

            # --- 2. Fairness (Lipschitz/Consistency) Penalty ---
            # We sample a batch of "fair pairs" to compute the penalty.
            # Some are sampled weighted by similarity, others are sampled uniformly.
            n_weighted = int(self.batch_size_pairs * self.weighted_frac)
            n_uniform = self.batch_size_pairs - n_weighted

            weighted_idxs = []
            if n_weighted > 0 and len(self.fair_pairs) > 0 and self._probs.sum() > 0:
                weighted_idxs = np.random.choice(len(self.fair_pairs), size=n_weighted, replace=False, p=self._probs)

            uniform_idxs = []
            if n_uniform > 0 and len(self.fair_pairs) > 0:
                remaining_indices = np.setdiff1d(np.arange(len(self.fair_pairs)), weighted_idxs, assume_unique=True)
                if len(remaining_indices) > 0:
                    uniform_idxs = np.random.choice(remaining_indices, size=min(n_uniform, len(remaining_indices)), replace=False)
            
            # Combine the indices for the full penalty batch.
            idxs_for_penalty = np.concatenate([weighted_idxs, uniform_idxs]).astype(int)

            penalty_term = torch.tensor(0.0, device=self.device)
            if len(idxs_for_penalty) > 0:
                # Loop through each sampled pair (i, j).
                for idx in idxs_for_penalty:
                    i, j, sim_ij = self.fair_pairs[idx]
                    
                    # Get the *initial, undrifted* states for both individuals in the pair.
                    # The fairness constraint is on the policy for the original states.
                    state_i_tensor = torch.tensor(self.all_dataset_states[i], dtype=torch.float32).unsqueeze(0).to(self.device)
                    state_j_tensor = torch.tensor(self.all_dataset_states[j], dtype=torch.float32).unsqueeze(0).to(self.device)
                    
                    # Get the Q-values for each state.
                    q_i = self.q_net(state_i_tensor)
                    q_j = self.q_net(state_j_tensor)

                    # Use softmax to convert Q-values into action probabilities. This represents the agent's policy.
                    prob_action1_s_i = torch.softmax(q_i, dim=1)[0, 1] # Probability of action 1 for person i
                    prob_action1_s_j = torch.softmax(q_j, dim=1)[0, 1] # Probability of action 1 for person j

                    # Calculate how different the policy is for these two individuals.
                    action_divergence = torch.abs(prob_action1_s_i - prob_action1_s_j)
                    
                    # Weight this divergence by the similarity of the pair.
                    # For very similar people (high sim_ij), we want the divergence to be very small.
                    penalty_term += sim_ij * action_divergence
                
                # Average the penalty over the batch of pairs.
                penalty = penalty_term / len(idxs_for_penalty)
            else:
                penalty = torch.tensor(0.0, device=self.device)

            # --- 3. Combine Losses and Update the Network ---
            # The total loss is the DQN loss plus our weighted fairness penalty.
            loss = loss_q + self.lambda_fair * penalty

            # Perform the standard backpropagation and optimization step.
            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # --- Logging ---
        # Keep track of training progress and log the different parts of the loss.
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss_q", loss_q.item())
        if self.lambda_fair > 0 and len(idxs_for_penalty) > 0:
            self.logger.record("train/penalty_lipschitz", penalty.item())
        self.logger.record("train/loss_total", loss.item())

# --- Fairness Evaluation Callback ---
# A callback is a special tool in Stable Baselines3 that lets us run custom code
# at specific points during the agent's training. This one is designed to
# periodically stop and evaluate our agent on a range of fairness and
# performance metrics.
class FairnessEvalCallback(BaseCallback):
    # A custom callback to evaluate the agent on various fairness metrics
    # at regular intervals during training.
    def __init__(
        self,
        full_df: pd.DataFrame,
        initial_undrifted_states: np.ndarray,       # The original dataset states, before any changes.
        fair_pairs_list: list,                      # The list of similar pairs we found earlier.
        sensitive_feature_name: str,                # The column name of the sensitive attribute (e.g., 'sex').
        all_feature_names_ordered_list: list,       # List of all feature names in their correct order.
        driftable_feature_names_list: list[str],    # List of features that can change.
        eval_env_creator,                           # A function that creates a new evaluation environment.
        eval_freq: int = 2000,                      # How often (in training steps) to run this evaluation.
        n_eval_episodes: int = 100,                 # Number of episodes to run for reward evaluation.
        sigma_final: float = 1.0,                   # A parameter for finding similar pairs in the *final* states.
        k_final: int = 5,                           # How many neighbors to look for in the *final* states.
        batch_size_pairs: int = 64,                 # Number of pairs to sample for final state consistency checks.
        verbose: int = 1                            # Set to 1 to print logs.
    ):
        super().__init__(verbose)
        # --- Store all the initial settings ---
        self.full_df = full_df
        self.initial_undrifted_states = initial_undrifted_states
        self.fair_pairs = fair_pairs_list
        self.sensitive_feature_name = sensitive_feature_name
        self.all_feature_names_ordered = all_feature_names_ordered_list
        self.driftable_features = driftable_feature_names_list
        # We need to be able to create fresh environments for evaluation to avoid any state carrying over.
        self.eval_env_creator = eval_env_creator
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.sigma_final = sigma_final
        self.k_final = k_final
        self.metrics_log = []

        # Pre-calculate indices for efficiency
        self.sensitive_idx = self.all_feature_names_ordered.index(self.sensitive_feature_name)
        self.driftable_idxs = {feat: self.all_feature_names_ordered.index(feat) for feat in self.driftable_features}
        self.sensitive_groups = self.full_df[self.sensitive_feature_name].unique()

        
        # We need to be able to create fresh environments for evaluation to avoid any state carrying over.
        self.eval_env_creator = eval_env_creator
        self.eval_env_instance = self.eval_env_creator() # Create one instance for general use.
        
        self.eval_freq = eval_freq
        self.n_eval_episodes_reward = n_eval_episodes
        self.n_eval_episodes_group_fairness = n_eval_episodes

        self.sigma_final = sigma_final
        self.k_final = k_final
        self.batch_size_pairs = batch_size_pairs

        # This list will hold the dictionary of metrics from each evaluation run.
        self.metrics_log = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        current_metrics = {"step": self.num_timesteps}
        print(f"\n--- Running Evaluation @ Timestep {self.num_timesteps} ---")

        ### 1. Individual Fairness on Initial (pre-drift) States
        actions_on_initial, _ = self.model.predict(self.initial_undrifted_states, deterministic=True)
        
        # Counterfactual Fairness (CFD)
        current_metrics["indiv_fairness_cfd_initial"] = self.calculate_cfd(self.initial_undrifted_states)
        
        # Consistency
        num, den = 0.0, 0.0
        for i, j, sim_ij in self.fair_pairs:
            num += sim_ij * abs(actions_on_initial[i] - actions_on_initial[j])
            den += sim_ij
        current_metrics["indiv_fairness_consistency_initial"] = 1.0 - (num / (den + 1e-8))

        # Lipschitz training penalty
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            # The correct way to get the most recent values is via the .name_to_value dictionary
            current_log_values = self.model.logger.name_to_value
            avg_lipschitz_penalty = current_log_values.get("train/penalty_lipschitz", float('nan'))
        else:
            avg_lipschitz_penalty = float('nan') # Fallback if logger is not available

        current_metrics["avg_lipschitz_penalty_train"] = avg_lipschitz_penalty

        ### 2. Combined Rollout for Performance, Drift, and Final States 
        temp_eval_env = self.eval_env_creator()
        
        # Initialize collectors
        successes_by_group = {group: 0 for group in self.sensitive_groups}
        counts_by_group = {group: 0 for group in self.sensitive_groups}
        drifts_by_group = {group: {feat: [] for feat in self.driftable_features} for group in self.sensitive_groups}
        final_states = []

        for _ in range(self.n_eval_episodes):
            obs, info = temp_eval_env.reset()
            sampled_idx = info["sampled_row_index"]
            group = self.full_df.loc[sampled_idx, self.sensitive_feature_name]
            
            done = False
            total_reward = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = temp_eval_env.step(action.item())
                total_reward += reward
                done = terminated or truncated

            ## Collect results from the completed episode
            # a) Performance (Success Rate)
            counts_by_group[group] += 1
            if total_reward > 0: # Success is a positive total reward
                successes_by_group[group] += 1
            
            # b) Final State
            final_states.append(obs)
            
            # c) Feature Drift
            for feat, col_idx in self.driftable_idxs.items():
                orig_val = self.initial_undrifted_states[sampled_idx, col_idx]
                final_val = obs[col_idx]
                pct_change = ((final_val - orig_val) / abs(orig_val) * 100.0) if orig_val != 0 else 0.0
                drifts_by_group[group][feat].append(pct_change)
        
        temp_eval_env.close()

        ### 3. Calculate and Log Metrics from Rollouts

        ## Performance Metrics 
        total_successes = sum(successes_by_group.values())
        total_episodes = sum(counts_by_group.values())
        current_metrics["perf_success_rate_overall"] = total_successes / total_episodes if total_episodes > 0 else 0.0
        
        for group in self.sensitive_groups:
            group_count = counts_by_group[group]
            group_successes = successes_by_group[group]
            current_metrics[f"perf_success_rate_group_{group}"] = group_successes / group_count if group_count > 0 else 0.0

        ## Drift Metrics
        for group in self.sensitive_groups:
            for feat in self.driftable_features:
                avg_drift = np.mean(drifts_by_group[group][feat]) if drifts_by_group[group][feat] else 0.0
                current_metrics[f"drift_pct_{feat.replace('.', '_')}_group_{group}"] = avg_drift

        ## Individual Fairness on Final States
        final_states_np = np.vstack(final_states)
        current_metrics["indiv_fairness_cfd_final"] = self.calculate_cfd(final_states_np)
        current_metrics["indiv_fairness_consistency_final"] = self.calculate_consistency(final_states_np)
        
        # Log all metrics to SB3 logger and internal log
        self.metrics_log.append(current_metrics)
        for key, value in current_metrics.items():
            if key != "step":
                self.logger.record(f"eval/{key}", value)
        
        return True

    def calculate_cfd(self, states_to_eval):
        cf_states = states_to_eval.copy()
        cf_states[:, self.sensitive_idx] = 1 - cf_states[:, self.sensitive_idx]
    
        acts_orig, _ = self.model.predict(states_to_eval, deterministic=True)
        acts_cf, _ = self.model.predict(cf_states, deterministic=True)
        return float(np.mean(np.abs(acts_orig - acts_cf)))

    def calculate_consistency(self, states_to_eval):
        if len(states_to_eval) < self.k_final + 1:
            return 1.0 # Not enough samples to find neighbors

        k = self.k_final + 1 # +1 to account for the point itself
        nbrs = NearestNeighbors(n_neighbors=k).fit(states_to_eval)
        dists, idxs = nbrs.kneighbors(states_to_eval)
        sims = np.exp(-(dists**2) / (self.sigma_final**2))
    
        actions_final, _ = self.model.predict(states_to_eval, deterministic=True)
    
        num_c, den_c = 0.0, 0.0
        for i in range(len(states_to_eval)):
            for rank in range(1, k): # Skip the point itself
                j = idxs[i, rank]
                sim_ij = sims[i, rank]
                diff = abs(int(actions_final[i]) - int(actions_final[j]))
                num_c += sim_ij * diff
                den_c += sim_ij
    
        return 1.0 - (num_c / (den_c + 1e-8))

    def get_metrics_log_df(self):
        return pd.DataFrame(self.metrics_log)

### Plotting function for the metrics
def plot_results(metrics_df, save_path, config):
    print("Generating final plots...")
    plt.figure(figsize=(18, 10))
    
    # Plot 1: Main Performance - Per-Group Success Rate
    ax = plt.subplot(2, 3, 1)
    sensitive_groups = [0, 1] # Assuming binary groups for plotting
    for group in sensitive_groups:
        ax.plot(metrics_df["step"], metrics_df[f"perf_success_rate_group_{group}"], label=f"Success Rate (Group {group})")
    ax.plot(metrics_df["step"], metrics_df["perf_success_rate_overall"], label="Overall Success Rate", linestyle='--', color='black', alpha=0.7)
    ax.set_title("Agent Performance & Outcome Fairness")
    ax.set_xlabel("Training Timesteps")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Individual Fairness - Consistency
    ax = plt.subplot(2, 3, 2)
    ax.plot(metrics_df["step"], metrics_df["indiv_fairness_consistency_initial"], label="Initial States")
    ax.plot(metrics_df["step"], metrics_df["indiv_fairness_consistency_final"], label="Final States")
    ax.set_title("Individual Fairness: Consistency")
    ax.set_xlabel("Training Timesteps")
    ax.set_ylabel("Consistency Score (1 is best)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Individual Fairness - Counterfactual Fairness (CFD)
    ax = plt.subplot(2, 3, 3)
    ax.plot(metrics_df["step"], metrics_df["indiv_fairness_cfd_initial"], label="Initial States")
    ax.plot(metrics_df["step"], metrics_df["indiv_fairness_cfd_final"], label="Final States")
    ax.set_title("Individual Fairness: CFD")
    ax.set_xlabel("Training Timesteps")
    ax.set_ylabel("CFD (0 is best)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plots 4 & 5: Feature Drift
    driftable_features = config['dataset_columns']['driftable_features']
    for i, feat in enumerate(driftable_features):
        if i >= 2: break # Limit to 2 drift plots for a 2x3 grid
        ax = plt.subplot(2, 3, 4 + i)
        safe_feat_name = feat.replace('.', '_')
        for group in sensitive_groups:
             ax.plot(metrics_df["step"], metrics_df[f"drift_pct_{safe_feat_name}_group_{group}"], label=f"Group {group}")
        ax.set_title(f"Feature Drift: {feat}")
        ax.set_xlabel("Training Timesteps")
        ax.set_ylabel("Average % Change")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 6: The Training Fairness Penalty
    ax = plt.subplot(2, 3, 6)
    ax.plot(metrics_df["step"], metrics_df["avg_lipschitz_penalty_train"], color='purple')
    ax.set_title("Training Fairness Penalty (λ_fair)")
    ax.set_xlabel("Training Timesteps")
    ax.set_ylabel("Penalty Value")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plots saved to {save_path}")
    plt.show()

# --- Main Training Script ---
# This is the main part of our program where everything gets executed.
# We'll load the data, set up the agent and environment, run the training,
# and finally, plot the results to see how our fair agent learned.
if __name__ == '__main__':
    ### 0. Load Configuration from YAML
    print("Loading configuration from config.yaml...")
    with open('config_one_run.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Extract config sections for easier access
    paths = config['file_paths']
    cols = config['dataset_columns']
    h_params = config['fair_dqn_params']
    train_cfg = config['training']
    eval_cfg = config['evaluation']
    knn_cfg = config['knn_settings']
    reward_cfg = config['rewards']


    ### 1. Setup and Preprocessing
    # Set a random seed everywhere for reproducibility. This ensures that if we
    # run the script again, we get the exact same results.
    SEED = train_cfg['seed']
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    print("Loading and preprocessing data...")
    df = pd.read_csv(paths['input_data'])

    # Target column
    GROUND_TRUTH_TARGET_COL = f"{cols['target_original']}_binary"
    df[GROUND_TRUTH_TARGET_COL] = (df[cols['target_original']] == cols['target_positive_class']).astype(int)

    # We define which features can change over time (driftable) and which cannot (static).
    DRIFTABLE_FEATURES = cols['driftable_features']
    potential_features = [c for c in df.columns if c not in [cols['target_original'], GROUND_TRUTH_TARGET_COL]]
    STATIC_FEATURES = [c for c in potential_features if c not in DRIFTABLE_FEATURES]
    
    # It's crucial to have a fixed order for features to feed into the models.
    ALL_FEATURE_NAMES_ORDERED = STATIC_FEATURES + DRIFTABLE_FEATURES
    
    # Convert our data into a NumPy array, which is what the RL models expect.
    states_np_array = df[ALL_FEATURE_NAMES_ORDERED].values.astype(np.float32)

    ### 2. Pre-compute Similar Pairs for Fairness 
    print("Pre-computing fairness pairs using k-NN...")
    nbrs = NearestNeighbors(n_neighbors=knn_cfg['k_neighbors'], algorithm='auto').fit(states_np_array)
    dists, idxs = nbrs.kneighbors(states_np_array)
    # Convert distances into similarities (closer distance -> higher similarity).
    sims = np.exp(-(dists**2) / (knn_cfg['sigma']**2))

    # Store all pairs and their similarities in a list for the agent and callback to use.
    fair_pairs = []
    for i in range(states_np_array.shape[0]):
        for neighbor_rank in range(knn_cfg['k_neighbors']):
            j = idxs[i, neighbor_rank]
            sim_ij = float(sims[i, neighbor_rank])
            fair_pairs.append((i, j, sim_ij))
    
    
    ### 4. Setup the Environment, Agent, and Callback
    print("Setting up RL environment, FairDQN agent, and callback...")
    # Define which feature we consider sensitive for our fairness metrics.
    SENSITIVE_ATTRIBUTE_NAME = cols['sensitive_attribute']
    
    
    # We create a function to generate our custom environment. 
    def create_env_fn():
        return EpisodicFairnessEnv(
            full_dataset_df=df.copy(),
            static_feature_names_list=STATIC_FEATURES,
            driftable_feature_names_list=DRIFTABLE_FEATURES,
            ordered_feature_names_list=ALL_FEATURE_NAMES_ORDERED,
            ground_truth_target_column=GROUND_TRUTH_TARGET_COL,
            drift_logic=config['drift_logic'],
            max_episode_steps=train_cfg['max_episode_steps'],
            terminal_reward=reward_cfg['terminal_reward'],
            terminal_penalty=reward_cfg['terminal_penalty'],
            step_cost=reward_cfg['step_cost']
        )

    # Create the main training environment.
    train_env = create_env_fn()
    train_env.reset(seed=SEED)
    
    # Automatically use the GPU if it's available, or according to config.
    device = train_cfg['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize our custom FairDQN agent with all its parameters.
    # `lambda_fair` is the key one that controls the strength of the fairness penalty.
    # Initialize our custom FairDQN agent with all its parameters.
    # The first set of parameters are custom to our FairDQN for the fairness logic.
    # The rest are standard hyperparameters for the Stable Baselines3 DQN model.
    # Pass all hyperparameters from the config dictionary
    # Check config dictionary for more info on each parameter.
    model = FairDQN(
        "MlpPolicy",
        train_env,
        all_dataset_states=states_np_array,
        fair_pairs_list=fair_pairs,
        seed=SEED,
        verbose=0,
        device=device,
        **h_params # Pass all DQN hyperparameters with the ** operator from config
    )

    # Initialize our evaluation callback, giving it all the data it needs to calculate metrics.
    callback = FairnessEvalCallback(
        full_df=df,
        initial_undrifted_states=states_np_array,
        fair_pairs_list=fair_pairs,
        sensitive_feature_name=SENSITIVE_ATTRIBUTE_NAME,
        all_feature_names_ordered_list=ALL_FEATURE_NAMES_ORDERED,
        driftable_feature_names_list=DRIFTABLE_FEATURES,
        eval_env_creator=create_env_fn,
        sigma_final=knn_cfg['sigma'],
        k_final=knn_cfg['k_neighbors'],
        verbose=1,
        **eval_cfg
    )

    ### 5. Train the Agent 
    print("\nStarting training... Evaluation metrics will be printed periodically.")
    # This kicks off the training process. The callback will be automatically
    # called by the `.learn()` method at the specified frequency.
    model.learn(total_timesteps=train_cfg['total_timesteps'], callback=callback)
    print("\nTraining finished.")

    ### 6. Process and Visualize Results 
    # Now that training is done, we get the metrics from our callback and plot them.
    metrics_df = callback.get_metrics_log_df()
    # We can ignore the early evaluations before the agent really started learning.
    metrics_df = metrics_df[metrics_df["step"] > model.learning_starts].reset_index(drop=True)
    # Set pandas display options to see the full DataFrame.
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print("\n--- Evaluation Metrics Over Time ---")
    print(metrics_df)
    # Save the raw metrics to a CSV file for analysis.
    metrics_df.to_csv(paths['metrics_output_csv'], index=False)

    ## Call plotting function
    plot_results(metrics_df, paths['plot_output_image'], config)

    # Clean up the environment.
    train_env.close()
