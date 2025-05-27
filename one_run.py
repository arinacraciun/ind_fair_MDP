# Import all the necessary libraries. We're bringing in tools for building the reinforcement
# learning environment (gymnasium), handling data (numpy, pandas), finding similar
# neighbors (sklearn), building and managing the RL agent (stable_baselines3, torch),
# and a few other utilities for plotting, saving models, suppressing warnings and configs.
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


# --- Proxy Model Functions ---
# This section contains helper functions to train and load a "proxy model".
# This model learns to predict an outcome based on the non-changeable (static)
# features of an individual. Our RL agent will then use this proxy model's
# predictions as the "correct" action to take in the environment.

def train_and_save_proxy_model(
    df_train: pd.DataFrame, # The training data, including features and the target label.
    static_feature_names_list: list[str], # A list of column names for the static features.
    target_column_name_proxy: str, # The name of the column we want the model to predict.
    model_save_path: str = "proxy_logistic_model.joblib" # Where to save the trained model.
):
    """
    Trains a simple Logistic Regression model on static features and saves it to a file.
    This model acts as a stand-in for a real-world decision-making system.
    """
    print(f"Training proxy model with static features: {static_feature_names_list}")
    # We'll use only the static features to train this model.
    X_static = df_train[static_feature_names_list].copy()
    y = df_train[target_column_name_proxy].copy()

    # Initialize and train a standard logistic regression model.
    proxy_model = LogisticRegression(solver='liblinear', random_state=42)
    proxy_model.fit(X_static, y)

    # We see a quick example of its predictions.
    print(f"Proxy model trained. Example prediction for first few static samples: {proxy_model.predict(X_static.head())}")
    # Save the trained model to the specified path for later use.
    joblib.dump(proxy_model, model_save_path)
    print(f"Proxy model saved to {model_save_path}")
    return proxy_model

def get_or_train_proxy_model(
    model_path: str,
    df_train: pd.DataFrame,
    static_feature_names_list: list[str],
    target_column_name_proxy: str
) -> LogisticRegression:
    """
    The main manager function for the proxy model.
    It first tries to load the model from `model_path`.
    If the file doesn't exist, it automatically calls the training function.
    """
    # First, try to load the model from the specified path.
    try:
        if os.path.exists(model_path):
            print(f"Found existing proxy model. Loading from: {model_path}")
            proxy_model = joblib.load(model_path)
            print("Proxy model loaded successfully.")
            return proxy_model
        else:
            # This 'else' will trigger the exception, making the logic flow to the 'except' block.
            # This is a clean way to handle the "file not found" case.
            raise FileNotFoundError

    # If the file doesn't exist, this block will execute.
    except FileNotFoundError:
        print(f"Proxy model not found at '{model_path}'.")
        # Call the training function to create, save, and return a new model.
        return train_and_save_proxy_model(
            df_train=df_train,
            static_feature_names_list=static_feature_names_list,
            target_column_name_proxy=target_column_name_proxy,
            model_save_path=model_path
        )

# --- Episodic Fairness Environment ---
# This is our custom reinforcement learning environment. An "episode" in this
# environment consists of the agent making a series of decisions for a single
# individual sampled from the dataset.
class EpisodicFairnessEnv(gym.Env):
    """
    A custom Gymnasium environment where an agent interacts with individuals
    from a dataset. The agent's actions can cause features of the individual
    to "drift" or change over time.
    """
    metadata = {'render_modes': [], 'render_fps': 4}

    def __init__(self,
                 full_dataset_df: pd.DataFrame,
                 static_feature_names_list: list[str],
                 driftable_feature_names_list: list[str],
                 ordered_feature_names_list: list[str], # To keep the observation data consistent.
                 target_column_name_for_proxy_training: str,
                 proxy_model: str = "proxy_logistic_model.joblib",
                 max_episode_steps: int = 10, # How many decisions to make per individual.
                 drift_logic: dict = None): # Settings for how features change.
        super().__init__()

        # --- Basic Environment Setup ---
        self.full_dataset = full_dataset_df.reset_index(drop=True)
        self.static_feature_names = static_feature_names_list
        self.driftable_feature_names = driftable_feature_names_list
        self.all_feature_names_ordered = ordered_feature_names_list
        self.target_column_name_for_proxy = target_column_name_for_proxy_training
        self.max_episode_steps = max_episode_steps
        self.proxy_model = proxy_model

        # Store the generic drift logic
        self.drift_logic = drift_logic if drift_logic is not None else {}

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
        self.current_Y_proxy = None              # The proxy model's prediction for this person.
        self.current_original_target_from_df = None # The actual original outcome from the dataset.
        self.episode_step_count = 0
        self.np_random = None # The random number generator for reproducibility.

    def _get_observation(self) -> np.ndarray:
        """
        Constructs the observation vector for the agent. It combines the static
        and current (possibly drifted) driftable features in a consistent order.
        """
        # Combine the feature sets into a single DataFrame.
        obs_data = pd.concat([
            self.current_static_features_df[self.static_feature_names],
            self.current_driftable_features_df[self.driftable_feature_names]
        ], axis=1)
        # Ensure the final order is correct and return as a numpy array.
        return obs_data[self.all_feature_names_ordered].values.flatten().astype(np.float32)

    def _apply_drift(self, action: int):
        """
        Modifies the driftable features based on the agent's action. This simulates
        the real-world consequence of a decision.
        """
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
        """
        Starts a new episode. It picks a random person from the dataset and
        initializes the state.
        """
        super().reset(seed=seed) # This is important for seeding the random number generator.

        # Randomly sample one individual from the full dataset for this episode.
        self.current_row_index = self.np_random.integers(0, len(self.full_dataset))
        # Make a copy of this individual's data so we can modify it without affecting the original dataset.
        current_individual_data_for_episode = self.full_dataset.iloc[[self.current_row_index]].copy()

        # Separate the static and driftable features for this individual.
        self.current_static_features_df = current_individual_data_for_episode[self.static_feature_names]
        self.current_driftable_features_df = current_individual_data_for_episode

        # Use the proxy model to get a "target" prediction based on the initial static features.
        # This prediction will remain constant throughout the episode.
        static_features_for_proxy_np = self.current_static_features_df.values
        self.current_Y_proxy = self.proxy_model.predict(static_features_for_proxy_np)[0]

        # Store the original target label from the dataset for analysis.
        self.current_original_target_from_df = current_individual_data_for_episode[self.target_column_name_for_proxy].iloc[0]
        self.episode_step_count = 0

        # Get the initial observation and create an info dictionary with helpful metadata.
        observation = self._get_observation()
        info = {
            "Y_proxy": self.current_Y_proxy,
            "original_target": self.current_original_target_from_df,
            "sampled_row_index": self.current_row_index
        }
        return observation, info

    def step(self, action: int):
        """
        Executes one time step within the episode.
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Action space is {self.action_space}")

        # The reward is simple: +1 for matching the proxy model's prediction, -1 for not.
        # This encourages the agent to learn the proxy model's logic.
        reward = 1.0 if action == self.current_Y_proxy else -1.0

        # Apply the feature drift based on the action taken.
        self._apply_drift(action)
        # Get the new state of the environment.
        observation = self._get_observation()

        self.episode_step_count += 1
        # The episode ends if we've reached the maximum number of steps.
        terminated = self.episode_step_count >= self.max_episode_steps
        truncated = False # We're not using a separate time limit truncation.

        # Pack up useful info about this step.
        info = {
            "Y_proxy": self.current_Y_proxy,
            "original_target": self.current_original_target_from_df,
            "action_taken": action,
            "reward_received": reward,
            "sampled_row_index": self.current_row_index
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
    """
    A custom callback to evaluate the agent on various fairness metrics
    at regular intervals during training.

    It measures:
    1.  Standard performance (mean reward).
    2.  Fairness on the initial, unchanged dataset (CFD, Consistency).
    3.  Group fairness after running full episodes (Demographic Parity, Equalized Odds).
    4.  How much the agent's actions cause features to drift for different groups.
    5.  Fairness on the final, drifted states of individuals (CFD, Consistency after drift).
    """
    def __init__(
        self,
        full_df: pd.DataFrame,
        initial_undrifted_states: np.ndarray,       # The original dataset states, before any changes.
        fair_pairs_list: list,                      # The list of similar pairs we found earlier.
        true_labels_initial: np.ndarray,            # The original, true outcomes for each person.
        sensitive_feature_name: str,                # The column name of the sensitive attribute (e.g., 'sex').
        all_feature_names_ordered_list: list,       # List of all feature names in their correct order.
        driftable_feature_names_list: list[str],    # List of features that can change.
        eval_env_creator,                           # A function that creates a new evaluation environment.
        eval_freq: int = 2000,                      # How often (in training steps) to run this evaluation.
        n_eval_episodes_reward: int = 5,            # Number of episodes to run for standard reward evaluation.
        n_eval_episodes_group_fairness: int = 20,   # More episodes for robust fairness stats.
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
        self.true_labels_initial = true_labels_initial
        self.sensitive_feature_name = sensitive_feature_name
        self.all_feature_names_ordered = all_feature_names_ordered_list
        self.driftable_features = driftable_feature_names_list

        # For efficiency, we figure out the column index of our sensitive and driftable features.
        try:
            self.sensitive_idx_in_initial_states = self.all_feature_names_ordered.index(self.sensitive_feature_name)
        except ValueError:
            raise ValueError(f"Sensitive feature '{self.sensitive_feature_name}' not in 'all_feature_names_ordered_list'. Check names.")

        self.driftable_idxs = {
            feat: self.all_feature_names_ordered.index(feat)
            for feat in self.driftable_features
        }
        
        # We need to be able to create fresh environments for evaluation to avoid any state carrying over.
        self.eval_env_creator = eval_env_creator
        self.eval_env_instance = self.eval_env_creator() # Create one instance for general use.
        
        self.eval_freq = eval_freq
        self.n_eval_episodes_reward = n_eval_episodes_reward
        self.n_eval_episodes_group_fairness = n_eval_episodes_group_fairness

        self.sigma_final = sigma_final
        self.k_final = k_final
        self.batch_size_pairs = batch_size_pairs

        # This list will hold the dictionary of metrics from each evaluation run.
        self.metrics_log = []

    def _on_step(self) -> bool:
        """
        This function is called by the trainer after every step. We only run our
        full evaluation logic every `eval_freq` steps.
        """
        # Check if it's time to run the evaluation.
        if self.num_timesteps % self.eval_freq == 0:
            current_metrics = {"step": self.num_timesteps}

            # --- METRIC 1: Mean Reward (Standard Performance) ---
            # How well is the agent doing at its main task of matching the proxy model?
            temp_eval_env_for_reward = self.eval_env_creator()
            mean_reward, _ = evaluate_policy(self.model, temp_eval_env_for_reward,
                        n_eval_episodes=self.n_eval_episodes_reward,
                        deterministic=True, render=False, warn=False)
            current_metrics["mean_reward_vs_Yproxy"] = mean_reward
            temp_eval_env_for_reward.close()

            # --- FAIRNESS ON INITIAL, UNCHANGED STATES ---
            # Here, we check the agent's fairness on the original data, before any drift happens.
            # This tells us if the learned policy itself is biased from the start.
            actions_on_initial_states, _ = self.model.predict(self.initial_undrifted_states, deterministic=True)

            # --- METRIC 2: Counterfactual Fairness (CFD) on Initial States ---
            # If we flip the sensitive attribute of a person (e.g., male to female),
            # how much does the agent's decision change? Ideally, it shouldn't change at all.
            cf_states = self.initial_undrifted_states.copy()
            cf_states[:, self.sensitive_idx_in_initial_states] = 1 - cf_states[:, self.sensitive_idx_in_initial_states]
            actions_cf, _ = self.model.predict(cf_states, deterministic=True)
            cfd = np.mean(np.abs(actions_on_initial_states - actions_cf))
            current_metrics["cfd_on_initial"] = cfd

            # --- METRIC 3: Consistency on Initial States ---
            # Do similar people get similar outcomes? We use our pre-computed `fair_pairs`
            # to measure this. A score of 1.0 is perfectly consistent.
            num, den = 0.0, 0.0
            for i, j, sim_ij in self.fair_pairs:
                num += sim_ij * abs(actions_on_initial_states[i] - actions_on_initial_states[j])
                den += sim_ij
            consistency = 1.0 - (num / (den + 1e-8)) if den > 1e-8 else 1.0
            current_metrics["consistency_on_initial"] = consistency
            
            # On the first evaluation, log the base accuracy of the proxy model we're learning from.
            if self.num_timesteps == self.eval_freq:
                proxy_preds_on_static = self.eval_env_instance.proxy_model.predict(
                    self.full_df[self.eval_env_instance.static_feature_names].values
                )
                proxy_accuracy = np.mean(proxy_preds_on_static == self.full_df[self.eval_env_instance.target_column_name_for_proxy])
                self.logger.record("eval/proxy_model_accuracy_on_full_data", proxy_accuracy)

            # Also grab the fairness penalty value from the last training batch.
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                current_log_values = self.model.logger.name_to_value
                avg_lipschitz_penalty = current_log_values.get("train/penalty_lipschitz", float('nan'))
            else:
                avg_lipschitz_penalty = float('nan') # Fallback if logger is not available
            current_metrics["avg_lipschitz_penalty_train"] = avg_lipschitz_penalty

            # --- GROUP FAIRNESS METRICS (from running full episodes) ---
            # Now we look at fairness not just on the initial state, but over the whole process.
            # We'll run several full episodes and collect the results.
            collected_actions_episodic = []
            collected_Y_proxy_episodic = []
            collected_sensitive_values_episodic = []
            
            temp_eval_env_for_group = self.eval_env_creator()
            for _ in range(self.n_eval_episodes_group_fairness):
                obs, info = temp_eval_env_for_group.reset()
                sampled_idx = info["sampled_row_index"]
                sensitive_val = self.full_df.loc[sampled_idx, self.sensitive_feature_name]
                
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, info_step = temp_eval_env_for_group.step(action.item())
                    
                    # For each step in the episode, save the action, target, and sensitive group.
                    collected_actions_episodic.append(action)
                    collected_Y_proxy_episodic.append(info_step['Y_proxy'])
                    collected_sensitive_values_episodic.append(sensitive_val)
                    done = terminated or truncated
            temp_eval_env_for_group.close()

            y_pred_ep = np.array(collected_actions_episodic)
            y_true_ep = np.array(collected_Y_proxy_episodic)
            sensitive_attr_ep = np.array(collected_sensitive_values_episodic)

            # --- METRIC 4: Demographic Parity (DP) Gap ---
            # Does the agent grant a positive outcome (action=1) at similar rates
            # across different groups?
            mask0_ep = (sensitive_attr_ep == 0)
            mask1_ep = (sensitive_attr_ep == 1)

            p0_ep = y_pred_ep[mask0_ep].mean() if mask0_ep.any() else 0.0
            p1_ep = y_pred_ep[mask1_ep].mean() if mask1_ep.any() else 0.0
            dp_gap_ep = abs(p0_ep - p1_ep)
            current_metrics["dp_gap_episodic"] = dp_gap_ep

            # --- METRIC 5: Equalized Odds (EO) Gaps ---
            # This is a stricter fairness metric. It checks if the agent's accuracy (both
            # True Positive Rate and False Positive Rate) is equal across groups.
            def get_tpr_fpr(y_true_group, y_pred_group):
                """Helper function to calculate TPR and FPR safely."""
                if len(y_true_group) == 0 or len(y_pred_group) == 0: return 0.0, 0.0
                # Use labels=[0,1] to ensure the confusion matrix has a consistent 2x2 shape.
                cm = confusion_matrix(y_true_group, y_pred_group, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()
                tpr = tp / (tp + fn + 1e-8) # True Positive Rate
                fpr = fp / (fp + tn + 1e-8) # False Positive Rate
                return tpr, fpr

            tpr0_ep, fpr0_ep = get_tpr_fpr(y_true_ep[mask0_ep], y_pred_ep[mask0_ep])
            tpr1_ep, fpr1_ep = get_tpr_fpr(y_true_ep[mask1_ep], y_pred_ep[mask1_ep])
            
            eo_tpr_gap_ep = abs(tpr0_ep - tpr1_ep)
            eo_fpr_gap_ep = abs(fpr0_ep - fpr1_ep)
            current_metrics["eo_tpr_gap_episodic"] = eo_tpr_gap_ep
            current_metrics["eo_fpr_gap_episodic"] = eo_fpr_gap_ep

            # --- METRIC 6: Feature Drift Analysis ---
            # Are the agent's actions causing features to change differently for different groups?
            # E.g., is one group's 'capital.gain' increasing more than the other's?
            drifts = {
                0: {feat: [] for feat in self.driftable_features}, # Group 0
                1: {feat: [] for feat in self.driftable_features}, # Group 1
            }

            env = self.eval_env_creator()
            for _ in range(self.n_eval_episodes_group_fairness):
                obs, info = env.reset()
                group = int(self.full_df.loc[info["sampled_row_index"], self.sensitive_feature_name])
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, info = env.step(action.item())
                    done = terminated or truncated

                # After the episode, compare the final feature values to the original ones.
                idx = info["sampled_row_index"]
                for feat, col_idx in self.driftable_idxs.items():
                    orig_val = self.initial_undrifted_states[idx, col_idx]
                    final_val = obs[col_idx]
                    # Calculate percentage change and store it.
                    pct_change = ((final_val - orig_val) / abs(orig_val) * 100.0) if orig_val != 0 else 0.0
                    drifts[group][feat].append(pct_change)
            env.close()

            # Average the drift percentages for each group and log them.
            for grp in [0, 1]:
                for feat in self.driftable_features:
                    avg_drift = float(np.mean(drifts[grp][feat])) if len(drifts[grp][feat]) else 0.0
                    key = f"drift_pct_{feat.replace('.','_')}_grp{grp}"
                    current_metrics[key] = avg_drift
                    self.logger.record(f"eval/{key}", avg_drift)

            # --- FAIRNESS ON FINAL, DRIFTED STATES ---
            # After the agent interacts with individuals, their states change.
            # We need to check if the agent is *still* fair on these new, drifted states.
            final_states = []
            env = self.eval_env_creator()
            for _ in range(self.n_eval_episodes_group_fairness):
                obs, info = env.reset()
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, term, trunc, info = env.step(action.item())
                    done = term or trunc
                final_states.append(obs)
            final_states = np.vstack(final_states)
            env.close()
            
            # --- METRIC 7: CFD on Final States ---
            cfd_final = self.calculate_cfd(final_states)
            current_metrics["cfd_on_final"] = cfd_final
            self.logger.record("eval/cfd_on_final", cfd_final)
            
            # --- METRIC 8: Consistency on Final States ---
            # The original pairs might not be similar anymore. So, we find new neighbors
            # in the final state space and check for consistency there.
            final_consistency = self.calculate_consistency(final_states)
            current_metrics["consistency_on_final"] = final_consistency
            self.logger.record("eval/consistency_on_final", final_consistency)

            # --- Finalize and Log ---
            self.metrics_log.append(current_metrics)
            for key, value in current_metrics.items():
                if key != "step": self.logger.record(f"eval/{key}", value)
            
            if self.verbose > 0:
                print(f"--- Step {self.num_timesteps} Evaluation ---")
                for k, v in current_metrics.items():
                    if k != "step":
                        print(f"  {k}: {v:.4f}")

        return True

    def calculate_cfd(self, states_to_eval):
        """Helper to calculate Counterfactual Fairness on a given set of states."""
        sens_idx = self.sensitive_idx_in_initial_states
        cf_states = states_to_eval.copy()
        cf_states[:, sens_idx] = 1 - cf_states[:, sens_idx]
    
        acts_orig, _ = self.model.predict(states_to_eval, deterministic=True)
        acts_cf, _ = self.model.predict(cf_states, deterministic=True)
        return float(np.mean(np.abs(acts_orig - acts_cf)))

    def calculate_consistency(self, states_to_eval):
        """Helper to calculate Consistency on a given set of states."""
        # Find the k-nearest neighbors in this new state space.
        k = min(self.k_final + 1, len(states_to_eval))
        nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(states_to_eval)
        dists, idxs = nbrs.kneighbors(states_to_eval)
        # Convert distances to similarities.
        sims = np.exp(-(dists**2) / (self.sigma_final**2))
    
        # Get all pairs of neighbors.
        pairs = []
        for i in range(len(states_to_eval)):
            for rank in range(1, k): # Skip the first neighbor, which is the point itself.
                j = idxs[i, rank]
                sim_ij = sims[i, rank]
                pairs.append((i, j, sim_ij))
    
        # Get the agent's actions for all these final states.
        actions_final, _ = self.model.predict(states_to_eval, deterministic=True)
    
        num_c, den_c = 0.0, 0.0
        for i, j, sim_ij in pairs:
            diff = abs(int(actions_final[i]) - int(actions_final[j]))
            num_c += sim_ij * diff
            den_c += sim_ij
    
        return 1.0 - (num_c / (den_c + 1e-8))

    def get_metrics_log_df(self):
        """
        Returns all the collected metrics as a pandas DataFrame.
        """
        return pd.DataFrame(self.metrics_log)

# --- Main Training Script ---
# This is the main part of our program where everything gets executed.
# We'll load the data, set up the agent and environment, run the training,
# and finally, plot the results to see how our fair agent learned.
if __name__ == '__main__':
    # --- 0. Load Configuration from YAML ---
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
    drift_logic_cfg = config['drift_logic']


    # --- 1. Setup and Preprocessing ---
    # Set a random seed everywhere for reproducibility. This ensures that if we
    # run the script again, we get the exact same results.
    SEED = train_cfg['seed']
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    print("Loading and preprocessing data...")
    df = pd.read_csv("cleaned_adult.csv")

    # The original 'income' column is text. We'll create a binary 0/1 version
    # which is easier for models to work with. This will be the target our
    # proxy model tries to predict.
    TARGET_COLUMN_PROXY = f"{cols['target_original']}_proxy"
    df[TARGET_COLUMN_PROXY] = (df[cols['target_original']] == cols['target_positive_class']).astype(int)

    # We define which features can change over time (driftable) and which cannot (static).
    DRIFTABLE_FEATURES = cols['driftable_features']
    potential_features = [c for c in df.columns if c not in [cols['target_original'], TARGET_COLUMN_PROXY]]
    STATIC_FEATURES = [c for c in potential_features if c not in DRIFTABLE_FEATURES]
    
    # It's crucial to have a fixed order for features to feed into the models.
    ALL_FEATURE_NAMES_ORDERED = STATIC_FEATURES + DRIFTABLE_FEATURES
    
    # Convert our data into a NumPy array, which is what the RL models expect.
    states_np_array = df[ALL_FEATURE_NAMES_ORDERED].values.astype(np.float32)
    true_labels_for_initial_states = df[TARGET_COLUMN_PROXY].values

    # --- 2. Pre-compute Similar Pairs for Fairness ---
    # To enforce fairness, we need to know which individuals in the dataset are
    # "similar" to each other *before* any training starts. We do this once
    # using K-Nearest Neighbors.
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
    
    # --- 3. Load or Train the Proxy Model ---
    # The proxy model simulates a real-world system. We'll try to load a pre-trained
    # one to save time. If it doesn't exist, we'll train it and save it for next time.
    print("Checking for proxy model...")
    proxy_model = get_or_train_proxy_model(
        model_path=paths['proxy_model'], # Get path from config
        df_train=df,
        static_feature_names_list=STATIC_FEATURES,
        target_column_name_proxy=TARGET_COLUMN_PROXY
    )
    
    # --- 4. Setup the Environment, Agent, and Callback ---
    print("Setting up RL environment, FairDQN agent, and callback...")
    # Define which feature we consider sensitive for our fairness metrics.
    SENSITIVE_ATTRIBUTE_NAME = cols['sensitive_attribute']
    
    
    # We create a function to generate our custom environment. This is good practice
    # as the evaluation callback will use it to create fresh envs.
    def create_env_fn():
        return EpisodicFairnessEnv(
            full_dataset_df=df.copy(),
            static_feature_names_list=STATIC_FEATURES,
            driftable_feature_names_list=DRIFTABLE_FEATURES,
            ordered_feature_names_list=ALL_FEATURE_NAMES_ORDERED,
            target_column_name_for_proxy_training=TARGET_COLUMN_PROXY,
            proxy_model=proxy_model,
            max_episode_steps=10,
            drift_logic=drift_logic_cfg
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
        true_labels_initial=true_labels_for_initial_states,
        sensitive_feature_name=SENSITIVE_ATTRIBUTE_NAME,
        all_feature_names_ordered_list=ALL_FEATURE_NAMES_ORDERED,
        driftable_feature_names_list=DRIFTABLE_FEATURES,
        eval_env_creator=create_env_fn,
        sigma_final=knn_cfg['sigma'],
        k_final=knn_cfg['k_neighbors'],
        batch_size_pairs=h_params['batch_size_pairs'],
        verbose=1,
        **eval_cfg # Pass all evaluation parameters
    )

    # --- 5. Train the Agent ---
    print("\nStarting training... Evaluation metrics will be printed periodically.")
    # This kicks off the training process. The callback will be automatically
    # called by the `.learn()` method at the specified frequency.
    model.learn(total_timesteps=train_cfg['total_timesteps'], callback=callback)
    
    print("\nTraining finished.")

    # --- 6. Process and Visualize Results ---
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
    
    # Visualize all the different metrics we tracked over the course of training.
    print("\nGenerating plots...")
    plt.figure(figsize=(24, 12))
    
    # Plot 1: Standard RL reward
    plt.subplot(2, 5, 1)
    plt.plot(metrics_df["step"], metrics_df["mean_reward_vs_Yproxy"], label="Mean Reward")
    plt.xlabel("Timestep"); plt.ylabel("Reward"); plt.legend(); plt.title("Mean Reward")

    # Plot 2: The fairness penalty value
    plt.subplot(2, 5, 2)
    plt.plot(metrics_df["step"], metrics_df["avg_lipschitz_penalty_train"], label="Lipschitz Penalty")
    plt.xlabel("Timestep"); plt.ylabel("Penalty"); plt.legend(); plt.title("Training Fairness Penalty")

    # Plot 3: Counterfactual Fairness (CFD)
    plt.subplot(2, 5, 3)
    plt.plot(metrics_df["step"], metrics_df["cfd_on_initial"], label="Initial States")
    plt.plot(metrics_df["step"], metrics_df["cfd_on_final"], label="Final States")
    plt.xlabel("Timestep"); plt.ylabel("CFD"); plt.legend(); plt.title("Counterfactual Fairness")
    
    # Plot 4: Consistency
    plt.subplot(2, 5, 4)
    plt.plot(metrics_df["step"], metrics_df["consistency_on_initial"], label="Initial States")
    plt.plot(metrics_df["step"], metrics_df["consistency_on_final"], label="Final States")
    plt.xlabel("Timestep"); plt.ylabel("Consistency"); plt.legend(); plt.title("Consistency")

    # Plot 5: Demographic Parity Gap
    plt.subplot(2, 5, 5)
    plt.plot(metrics_df["step"], metrics_df["dp_gap_episodic"], label="DP Gap")
    plt.xlabel("Timestep"); plt.ylabel("DP Gap"); plt.legend(); plt.title("Demographic Parity Gap")

    # Plot 6: Equalized Odds Gaps
    plt.subplot(2, 5, 6)
    plt.plot(metrics_df["step"], metrics_df["eo_tpr_gap_episodic"], label="TPR Gap")
    plt.plot(metrics_df["step"], metrics_df["eo_fpr_gap_episodic"], label="FPR Gap")
    plt.xlabel("Timestep"); plt.ylabel("EO Gap"); plt.legend(); plt.title("Equalized Odds Gaps")
    
    # Plot 7 & 8: TPR and FPR broken down by group
    plt.subplot(2, 5, 7)
    plt.plot(metrics_df["step"], metrics_df["tpr_group0_ep"], label="TPR Group 0")
    plt.plot(metrics_df["step"], metrics_df["tpr_group1_ep"], label="TPR Group 1")
    plt.xlabel("Timestep"); plt.ylabel("TPR"); plt.legend(); plt.title("True Positive Rates")

    plt.subplot(2, 5, 8)
    plt.plot(metrics_df["step"], metrics_df["fpr_group0_ep"], label="FPR Group 0")
    plt.plot(metrics_df["step"], metrics_df["fpr_group1_ep"], label="FPR Group 1")
    plt.xlabel("Timestep"); plt.ylabel("FPR"); plt.legend(); plt.title("False Positive Rates")

    # Plot 9 & 10: Feature drift for each group
    start_plot_index = 9
    for i, feat in enumerate(DRIFTABLE_FEATURES):
        plt.subplot(2, 5, start_plot_index + i) 
        safe_feat_name = feat.replace('.', '_')
        plt.plot(metrics_df["step"], metrics_df[f"drift_pct_{safe_feat_name}_grp0"], label="Group 0")
        plt.plot(metrics_df["step"], metrics_df[f"drift_pct_{safe_feat_name}_grp1"], label="Group 1")
        plt.xlabel("Timestep"); plt.ylabel("% Drift"); plt.legend()
        plt.title(f"Average % Drift: {feat}")
    
    plt.tight_layout()
    plt.savefig(paths['plot_output_image'])
    print(f"\nPlots saved to {paths['plot_output_image']}")
    plt.show()

    # Clean up the environment.
    train_env.close()
