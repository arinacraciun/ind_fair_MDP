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

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Proxy Model Functions
def train_and_save_proxy_model(
    df_train: pd.DataFrame, # Should contain static features (pre-scaled) and target
    static_feature_names_list: list[str],
    target_column_name_proxy: str,
    model_save_path: str = "proxy_logistic_model.joblib"
):
    print(f"Training proxy model with static features: {static_feature_names_list}")
    X_static = df_train[static_feature_names_list].copy()
    y = df_train[target_column_name_proxy].copy()

    proxy_model = LogisticRegression(solver='liblinear', random_state=42)
    proxy_model.fit(X_static, y)
    
    print(f"Proxy model trained. Example prediction for first few static samples: {proxy_model.predict(X_static.head())}")
    joblib.dump(proxy_model, model_save_path)
    print(f"Proxy model saved to {model_save_path}")
    return proxy_model

def load_proxy_model(model_load_path: str = "proxy_logistic_model.joblib"):
    try:
        proxy_model = joblib.load(model_load_path)
        print(f"Proxy model loaded from {model_load_path}")
        return proxy_model
    except FileNotFoundError:
        print(f"Error: Proxy model not found at path: {model_load_path}")
        return None

# --- Episodic Fairness Environment ---
class EpisodicFairnessEnv(gym.Env):
    metadata = {'render_modes': [], 'render_fps': 4}

    def __init__(self,
                 full_dataset_df: pd.DataFrame,
                 static_feature_names_list: list[str],
                 driftable_feature_names_list: list[str],
                 ordered_feature_names_list: list[str], # To maintain consistent observation order
                 target_column_name_for_proxy_training: str,
                 proxy_model: str = "proxy_logistic_model.joblib",
                 max_episode_steps: int = 10, # Shorter for testing, adjust as needed
                 drift_config: dict = None):
        super().__init__()

        self.full_dataset = full_dataset_df.reset_index(drop=True)
        self.static_feature_names = static_feature_names_list
        self.driftable_feature_names = driftable_feature_names_list
        self.all_feature_names_ordered = ordered_feature_names_list # For observation vector
        self.target_column_name_for_proxy = target_column_name_for_proxy_training
        self.max_episode_steps = max_episode_steps
        self.drift_config = drift_config if drift_config is not None else {
            'capital_gain_pct': 5.0,   # percentage to *increase* capital.gain per action=1
            'capital_loss_pct': 5.0    # percentage to *increase* capital.loss per action=0
        }
        
        self.proxy_model = proxy_model

        self.action_space = spaces.Discrete(2)
        num_features = len(self.all_feature_names_ordered)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32)

        self.current_row_index = -1
        self.current_static_features_df = None # For proxy model (expects DataFrame)
        self.current_driftable_features_df = None # Will evolve, DataFrame
        self.current_Y_proxy = None
        self.current_original_target_from_df = None
        self.episode_step_count = 0
        self.np_random = None # Will be initialized by super().reset(seed=seed)

    def _get_observation(self) -> np.ndarray:
        # Combine static and current driftable features IN THE DEFINED ORDER
        # current_static_features_df and current_driftable_features_df have the sampled individual's data
        obs_data = pd.concat([
            self.current_static_features_df[self.static_feature_names], # ensure correct order within static
            self.current_driftable_features_df[self.driftable_feature_names] # ensure correct order within driftable
        ], axis=1)
        # Ensure the final observation matches ALL_FEATURE_NAMES_ORDERED
        return obs_data[self.all_feature_names_ordered].values.flatten().astype(np.float32)


    def _apply_drift(self, action: int):
        cg_col = 'capital.gain'
        cl_col = 'capital.loss'

        # read percentages and convert to fractional
        cg_frac = self.drift_config.get('capital_gain_pct', 0.0) / 100.0
        cl_frac = self.drift_config.get('capital_loss_pct', 0.0) / 100.0

        # locate the single-row index in copied DataFrame
        i = self.current_row_index

        if action == 1 and cg_col in self.current_driftable_features_df.columns:
            current_val = self.current_driftable_features_df.at[i, cg_col]
            # multiply by (1 + pct)
            self.current_driftable_features_df.at[i, cg_col] = current_val * (1.0 + cg_frac)

        elif action == 0 and cl_col in self.current_driftable_features_df.columns:
            current_val = self.current_driftable_features_df.at[i, cl_col]
            self.current_driftable_features_df.at[i, cl_col] = current_val * (1.0 + cl_frac)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Handles self.np_random initialization

        self.current_row_index = self.np_random.integers(0, len(self.full_dataset))
        # Create a COPY of the individual's data for the episode to allow modification
        current_individual_data_for_episode = self.full_dataset.iloc[[self.current_row_index]].copy()
        
        # Static features for proxy model (from the copied data)
        self.current_static_features_df = current_individual_data_for_episode[self.static_feature_names]
        
        # Driftable features (this DataFrame slice will be modified by _apply_drift)
        self.current_driftable_features_df = current_individual_data_for_episode # _apply_drift will modify this

        # Y_proxy for this individual (based on initial static features, pre-scaled)
        static_features_for_proxy_np = self.current_static_features_df.values # Already scaled
        self.current_Y_proxy = self.proxy_model.predict(static_features_for_proxy_np)[0]
        
        self.current_original_target_from_df = current_individual_data_for_episode[self.target_column_name_for_proxy].iloc[0]
        self.episode_step_count = 0

        observation = self._get_observation()
        info = {
            "Y_proxy": self.current_Y_proxy,
            "original_target": self.current_original_target_from_df,
            "sampled_row_index": self.current_row_index
        }
        return observation, info

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Action space is {self.action_space}")

        reward = 1.0 if action == self.current_Y_proxy else -1.0
        self._apply_drift(action)
        observation = self._get_observation()

        self.episode_step_count += 1
        terminated = self.episode_step_count >= self.max_episode_steps
        truncated = False # Not using time limit truncation separately here

        info = {
            "Y_proxy": self.current_Y_proxy, # Stays constant for the episode
            "original_target": self.current_original_target_from_df,
            "action_taken": action,
            "reward_received": reward,
            "sampled_row_index": self.current_row_index # To fetch sensitive attribute if needed
        }
        return observation, reward, terminated, truncated, info

    def close(self):
        pass

    def render(self):
        pass

# --- FairDQN (Lipschitz/Consistency Penalty in train) ---
class FairDQN(BaseDQN):
    def __init__(self,
                 policy,
                 env,
                 all_dataset_states: np.ndarray, # Initial, undrifted states
                 fair_pairs_list: list,
                 lambda_fair: float = 1.0,
                 batch_size_pairs: int = 64,
                 weighted_frac: float = 0.5,
                 **kwargs):
        super().__init__(policy, env, **kwargs)
        self.fair_pairs = fair_pairs_list
        self.lambda_fair = lambda_fair
        self.batch_size_pairs = batch_size_pairs
        self.weighted_frac = weighted_frac
        self.all_dataset_states = all_dataset_states # Used for Lipschitz penalty

        self._sims = np.array([sim for (_, _, sim) in self.fair_pairs])
        self._probs = self._sims / (self._sims.sum() + 1e-8) # Add epsilon for stability

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer) # For schedulers

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            
            with torch.no_grad():
                next_q_values = self.q_net_target(replay_data.next_observations)
                next_q_values, _ = next_q_values.max(dim=1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones.float()) * self.gamma * next_q_values

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = torch.gather(current_q_values, dim=1, index=replay_data.actions.long())
            
            loss_q = F.mse_loss(current_q_values.squeeze(1), target_q_values)

            # --- Lipschitz/Consistency Penalty ---
            n_weighted = int(self.batch_size_pairs * self.weighted_frac)
            n_uniform = self.batch_size_pairs - n_weighted
            
            weighted_idxs = []
            if n_weighted > 0 and len(self.fair_pairs) > 0 and self._probs.sum() > 0:
                 weighted_idxs = np.random.choice(
                    len(self.fair_pairs), size=n_weighted, replace=False, p=self._probs
                )
            
            uniform_idxs = []
            if n_uniform > 0 and len(self.fair_pairs) > 0:
                remaining_indices = np.setdiff1d(np.arange(len(self.fair_pairs)), weighted_idxs, assume_unique=True)
                if len(remaining_indices) > 0:
                    uniform_idxs = np.random.choice(
                        remaining_indices, size=min(n_uniform, len(remaining_indices)), replace=False
                    )
            
            idxs_for_penalty = np.concatenate([weighted_idxs, uniform_idxs]).astype(int)
            
            penalty_term = torch.tensor(0.0, device=self.device)
            if len(idxs_for_penalty) > 0:
                for idx in idxs_for_penalty:
                    i, j, sim_ij = self.fair_pairs[idx]
                    # Use all_dataset_states which are the initial, undrifted states
                    state_i_tensor = torch.tensor(self.all_dataset_states[i], dtype=torch.float32).unsqueeze(0).to(self.device)
                    state_j_tensor = torch.tensor(self.all_dataset_states[j], dtype=torch.float32).unsqueeze(0).to(self.device)
                    
                    q_i = self.q_net(state_i_tensor)
                    q_j = self.q_net(state_j_tensor)

                    # Softmax for probabilities (policy output for Lipschitz)
                    prob_action1_s_i = torch.softmax(q_i, dim=1)[0, 1] # P(action=1 | s_i)
                    prob_action1_s_j = torch.softmax(q_j, dim=1)[0, 1] # P(action=1 | s_j)
                    
                    action_divergence = torch.abs(prob_action1_s_i - prob_action1_s_j)
                    penalty_term += sim_ij * action_divergence
                
                penalty = penalty_term / len(idxs_for_penalty)
            else:
                penalty = torch.tensor(0.0, device=self.device)

            loss = loss_q + self.lambda_fair * penalty
            
            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss_q", loss_q.item())
        if self.lambda_fair > 0 and len(idxs_for_penalty) > 0:
            self.logger.record("train/penalty_lipschitz", penalty.item())
        self.logger.record("train/loss_total", loss.item())

# --- Fairness Evaluation Callback ---
class FairnessEvalCallback(BaseCallback):
    def __init__(
        self,
        initial_undrifted_states: np.ndarray, # For CFD/Consistency on initial states
        fair_pairs_list: list,
        true_labels_initial: np.ndarray, # Original labels for initial_undrifted_states
        sensitive_feature_name: str,      # Name of sensitive column in the original DataFrame
        all_feature_names_ordered_list: list, # To find index of sensitive feature
        driftable_feature_names_list: list[str], 
        eval_env_creator, # Function to create a new eval env: lambda: EpisodicFairnessEnv(...)
        eval_freq: int = 2000,
        n_eval_episodes_reward: int = 5,
        n_eval_episodes_group_fairness: int = 20,
        sigma_final: float = 1.0,           # bandwidth for final‐state kNN
        k_final:    int   = 5,              # how many neighbors in final batch
        batch_size_pairs: int = 64,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.initial_undrifted_states = initial_undrifted_states
        self.fair_pairs = fair_pairs_list
        self.true_labels_initial = true_labels_initial
        self.sensitive_feature_name = sensitive_feature_name
        self.all_feature_names_ordered = all_feature_names_ordered_list
        try:
            self.sensitive_idx_in_initial_states = self.all_feature_names_ordered.index(self.sensitive_feature_name)
        except ValueError:
            raise ValueError(f"Sensitive feature '{self.sensitive_feature_name}' not in 'all_feature_names_ordered_list'. Check names.")
        self.driftable_features = driftable_feature_names_list
        self.driftable_idxs = {
            feat: self.all_feature_names_ordered.index(feat)
            for feat in self.driftable_features
        }
        self.eval_env_creator = eval_env_creator
        self.eval_env_instance = self.eval_env_creator()
        
        self.eval_freq = eval_freq
        self.n_eval_episodes_reward = n_eval_episodes_reward
        self.n_eval_episodes_group_fairness = n_eval_episodes_group_fairness

        self.sigma_final = sigma_final
        self.k_final     = k_final
        self.batch_size_pairs = batch_size_pairs

        self.metrics_log = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            current_metrics = {"step": self.num_timesteps}

            # 1) Mean reward (uses EpisodicFairnessEnv, reward based on Y_proxy)
            temp_eval_env_for_reward = self.eval_env_creator()
            mean_reward, _ = evaluate_policy(
                self.model, temp_eval_env_for_reward,
                n_eval_episodes=self.n_eval_episodes_reward,
                deterministic=True, render=False, warn=False
            )
            current_metrics["mean_reward_vs_Yproxy"] = mean_reward
            temp_eval_env_for_reward.close()
            m = {"step": self.num_timesteps}
            # --- Fairness on Initial, Undrifted States ---
            actions_on_initial_states, _ = self.model.predict(self.initial_undrifted_states, deterministic=True)

            # 2) CFD (on initial states)
            cf_states = self.initial_undrifted_states.copy()
            cf_states[:, self.sensitive_idx_in_initial_states] = 1 - cf_states[:, self.sensitive_idx_in_initial_states]
            actions_cf, _ = self.model.predict(cf_states, deterministic=True)
            cfd = np.mean(np.abs(actions_on_initial_states - actions_cf))
            current_metrics["cfd_on_initial"] = cfd

            # 3) Consistency (on initial states)
            num, den = 0.0, 0.0
            for i, j, sim_ij in self.fair_pairs:
                num += sim_ij * abs(actions_on_initial_states[i] - actions_on_initial_states[j])
                den += sim_ij
            consistency = 1.0 - (num / (den + 1e-8)) if den > 1e-8 else 1.0
            current_metrics["consistency_on_initial"] = consistency
            
            if self.num_timesteps == self.eval_freq : 
                proxy_preds_on_static = self.eval_env_instance.proxy_model.predict(
                    df[self.eval_env_instance.static_feature_names].values
                )
                proxy_accuracy = np.mean(proxy_preds_on_static == df[self.eval_env_instance.target_column_name_for_proxy])
                self.logger.record("eval/proxy_model_accuracy_on_full_data", proxy_accuracy)


            if hasattr(self.model, 'logger') and self.model.logger is not None:
                current_log_values = self.model.logger.name_to_value
                avg_lipschitz_penalty = current_log_values.get("train/penalty_lipschitz", float('nan'))
            else:
                avg_lipschitz_penalty = float('nan') # Fallback if logger is not available
            current_metrics["avg_lipschitz_penalty_train"] = avg_lipschitz_penalty

            # --- Group Fairness Metrics (DP, EO) from Episodic Rollouts ---
            collected_actions_episodic = []
            collected_Y_proxy_episodic = []
            collected_sensitive_values_episodic = []
            
            temp_eval_env_for_group = self.eval_env_creator()
            for ep_num in range(self.n_eval_episodes_group_fairness):
                obs, info = temp_eval_env_for_group.reset()
                sampled_idx = info["sampled_row_index"]
                sensitive_val = df.loc[sampled_idx, self.sensitive_feature_name]  
                
                terminated, truncated = False, False
                ep_steps = 0
                while not (terminated or truncated) and ep_steps < temp_eval_env_for_group.max_episode_steps :
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, info_step = temp_eval_env_for_group.step(action)
                    
                    collected_actions_episodic.append(action)
                    collected_Y_proxy_episodic.append(info_step['Y_proxy'])
                    collected_sensitive_values_episodic.append(sensitive_val)
                    ep_steps +=1
            temp_eval_env_for_group.close()

            y_pred_ep = np.array(collected_actions_episodic)
            y_true_ep = np.array(collected_Y_proxy_episodic) 
            sensitive_attr_ep = np.array(collected_sensitive_values_episodic)

            unique_sens_vals = np.unique(sensitive_attr_ep)
            if not all(val in [0, 1] for val in unique_sens_vals) and len(unique_sens_vals) <=2 :
                if len(unique_sens_vals) == 2:
                    print(f"Warning: Remapping sensitive attribute values {unique_sens_vals} to 0 and 1 for group fairness.")
                    map_to_0 = unique_sens_vals[0]
                    sensitive_attr_ep = (sensitive_attr_ep != map_to_0).astype(int)

            mask0_ep = (sensitive_attr_ep == 0)
            mask1_ep = (sensitive_attr_ep == 1)

            p0_ep = y_pred_ep[mask0_ep].mean() if mask0_ep.any() else 0.0
            p1_ep = y_pred_ep[mask1_ep].mean() if mask1_ep.any() else 0.0
            dp_gap_ep = abs(p0_ep - p1_ep)
            current_metrics["dp_gap_episodic"] = dp_gap_ep

            def get_tpr_fpr(y_true_group, y_pred_group):
                if len(y_true_group) == 0: return 0.0, 0.0
                labels_to_check = [0,1]
                if len(y_pred_group) == 0: return 0.0, 0.0 # Or float('nan'), float('nan')

                cm = confusion_matrix(y_true_group, y_pred_group, labels=labels_to_check)
                if cm.size < 4: # Should ideally not happen if labels=[0,1] is used.
                                # but as a safeguard if a group has no true instances of a class.
                    tn, fp, fn, tp = 0,0,0,0
                    if cm.shape == (1,1): # Only one value, means all are either TN or TP (if y_true_group had only one class)
                        pass # Keep tn,fp,fn,tp as 0,0,0,0 which might lead to 0/0.
                    else: # Fallback if cm is malformed.
                        return float('nan'), float('nan')


                # If cm.size is 4, ravel will work.
                try:
                    tn, fp, fn, tp = cm.ravel()
                except ValueError: # Should be caught by cm.size < 4, but as a double check
                    return float('nan'), float('nan')


                tpr = tp / (tp + fn + 1e-8)
                fpr = fp / (fp + tn + 1e-8)
                return tpr, fpr

            tpr0_ep, fpr0_ep = get_tpr_fpr(y_true_ep[mask0_ep], y_pred_ep[mask0_ep])
            tpr1_ep, fpr1_ep = get_tpr_fpr(y_true_ep[mask1_ep], y_pred_ep[mask1_ep])
            
            eo_tpr_gap_ep = abs(tpr0_ep - tpr1_ep)
            eo_fpr_gap_ep = abs(fpr0_ep - fpr1_ep)
            current_metrics["eo_tpr_gap_episodic"] = eo_tpr_gap_ep
            current_metrics["eo_fpr_gap_episodic"] = eo_fpr_gap_ep
            current_metrics["tpr_group0_ep"] = tpr0_ep
            current_metrics["fpr_group0_ep"] = fpr0_ep
            current_metrics["tpr_group1_ep"] = tpr1_ep
            current_metrics["fpr_group1_ep"] = fpr1_ep
            
            drifts = {
                0: {feat: [] for feat in self.driftable_features},
                1: {feat: [] for feat in self.driftable_features},
            }

            env = self.eval_env_creator()
            for _ in range(self.n_eval_episodes_group_fairness):
                obs, info = env.reset()
                group = int(df.loc[info["sampled_row_index"], self.sensitive_feature_name])
                # run until done
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                idx = info["sampled_row_index"]
                # for each driftable feature, compute % change vs original
                for feat, col_idx in self.driftable_idxs.items():
                    orig = self.initial_undrifted_states[idx, col_idx]
                    final = obs[col_idx]
                    if orig != 0:
                        pct = (final - orig) / abs(orig) * 100.0
                    else:
                        pct = 0.0
                    drifts[group][feat].append(pct)
            env.close()

            # Average and log the four new drift metrics
            for grp in [0, 1]:
                for feat in self.driftable_features:
                    arr = drifts[grp][feat]
                    avg = float(np.mean(arr)) if len(arr) else 0.0
                    key = f"drift_pct_{feat.replace('.','_')}_grp{grp}"
                    m[key] = avg
                    current_metrics[key] = avg
                    self.logger.record(f"eval/{key}", avg)

            final_states = []
            for ep in range(self.n_eval_episodes_group_fairness):
                obs, info = self.eval_env_instance.reset()
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, term, trunc, info = self.eval_env_instance.step(action)
                    done = term or trunc
                final_states.append(obs)
            final_states = np.vstack(final_states)  # shape (N_eps, D)

            # CFD on final (drifted) states
            sens_idx = self.all_feature_names_ordered.index(self.sensitive_feature_name)
            cf_final = final_states.copy()
            cf_final[:, sens_idx] = 1 - cf_final[:, sens_idx]

            acts_final,    _ = self.model.predict(final_states, deterministic=True)
            acts_final_cf, _ = self.model.predict(cf_final,     deterministic=True)
            cfd_final = float(np.mean(np.abs(acts_final - acts_final_cf)))
            current_metrics["cfd_on_final"] = cfd_final
            self.logger.record("eval/cfd_on_final", cfd_final)

            # Pairwise Q-value Lipschitz on final states 
            # build kNN on final_states
            k = min(self.k_final + 1, len(final_states))
            nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(final_states)
            dists, idxs = nbrs.kneighbors(final_states)
            sims = np.exp(-(dists**2) / (self.sigma_final**2))

            # collect pairs (skip self at rank 0)
            pairs = []
            for i in range(len(final_states)):
                for rank in range(1, k):
                    j = idxs[i, rank]
                    sim_ij = sims[i, rank]
                    pairs.append((i, j, sim_ij))

            # sample up to batch_size_pairs of them uniformly
            n_sample = min(self.batch_size_pairs, len(pairs))
            sel = np.random.choice(len(pairs), size=n_sample, replace=False)
            
            num, den = 0.0, 0.0
            for idx in sel:
                i, j, sim_ij = pairs[idx]
                s_i = torch.tensor(final_states[i], dtype=torch.float32).unsqueeze(0).to(self.model.device)
                s_j = torch.tensor(final_states[j], dtype=torch.float32).unsqueeze(0).to(self.model.device)

                q_i = self.model.q_net(s_i)[0]  # shape (n_actions,)
                q_j = self.model.q_net(s_j)[0]

                # maximum action‐value difference
                max_diff = float(torch.max(torch.abs(q_i - q_j)))
                num += sim_ij * max_diff
                den += sim_ij

            lips_final = float(num / (den + 1e-8))
            current_metrics["lipschitz_on_final"] = lips_final
            self.logger.record("eval/lipschitz_on_final", lips_final)


            # Consistency on final (drifted) states
            num_c, den_c = 0.0, 0.0
            for i, j, sim_ij in pairs:
                diff = abs(int(acts_final[i]) - int(acts_final[j]))
                num_c += sim_ij * diff
                den_c += sim_ij

            consistency_final = 1.0 - (num_c / (den_c + 1e-8))
            current_metrics["consistency_on_final"] = float(consistency_final)
            self.logger.record("eval/consistency_on_final", consistency_final)


            self.metrics_log.append(current_metrics)
            for key, value in current_metrics.items():
                if key != "step": self.logger.record(f"eval/{key}", value)
            
            if self.verbose > 0:
                print(f"--- Step {self.num_timesteps} Evaluation ---")
                for k, v_i in current_metrics.items(): # renamed v to v_i to avoid conflict
                    if k!="step" and isinstance(v_i, (int, float)): print(f"  {k}: {v_i}")
                    elif k!="step": print(f"  {k}: {v_i}")
        return True

    def get_metrics_log_df(self):
        return pd.DataFrame(self.metrics_log)

# --- Main Training Script ---
if __name__ == '__main__':
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # --- 1. Load and Preprocess Data ---
    df = pd.read_csv("cleaned_adult.csv")

    # Binarize the target column for the proxy model and 'true_labels'
    TARGET_COLUMN_ORIGINAL = 'income' 
    TARGET_COLUMN_PROXY = 'income_binary_gt_50k'
    df[TARGET_COLUMN_PROXY] = (df[TARGET_COLUMN_ORIGINAL] == '>50K').astype(int)

    DRIFTABLE_FEATURES = ['capital.gain', 'capital.loss']
    potential_features = [col for col in df.columns if col not in [TARGET_COLUMN_ORIGINAL, TARGET_COLUMN_PROXY]]
    STATIC_FEATURES = [col for col in potential_features if col not in DRIFTABLE_FEATURES]
    ALL_FEATURE_NAMES_ORDERED = STATIC_FEATURES + DRIFTABLE_FEATURES
    states_np_array = df[ALL_FEATURE_NAMES_ORDERED].values.astype(np.float32)
    true_labels_for_initial_states = df[TARGET_COLUMN_PROXY].values 

    # --- 2. Precompute kNN for fairness pairs (on initial states) ---
    k = 5
    sigma = 1.0 
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(states_np_array)
    dists, idxs = nbrs.kneighbors(states_np_array) 
    sims = np.exp(-(dists**2) / (sigma**2)) 

    fair_pairs = []
    for i in range(states_np_array.shape[0]):
        for neighbor_rank in range(k):
            j = idxs[i, neighbor_rank]
            sim_ij = float(sims[i, neighbor_rank])
            fair_pairs.append((i, j, sim_ij))
    
    # --- Proxy Model (Ensure it's trained once or loaded) ---
    proxy_model_path = "proxy_logistic_model.joblib"
    try:
        proxy_model = load_proxy_model(proxy_model_path)
    except FileNotFoundError:
        print("Proxy model not found, training a new one.")
        train_and_save_proxy_model(
            df, STATIC_FEATURES, TARGET_COLUMN_PROXY, proxy_model_path
        )


    SENSITIVE_ATTRIBUTE_NAME = 'sex' 
    if SENSITIVE_ATTRIBUTE_NAME not in df.columns:
        raise ValueError(f"Sensitive attribute '{SENSITIVE_ATTRIBUTE_NAME}' not found in DataFrame columns: {df.columns.tolist()}")
    if SENSITIVE_ATTRIBUTE_NAME not in ALL_FEATURE_NAMES_ORDERED:
            raise ValueError(f"Sensitive attribute '{SENSITIVE_ATTRIBUTE_NAME}' must be part of ALL_FEATURE_NAMES_ORDERED.")

    drift_settings = {
        'capital_gain_pct': 5.0,
        'capital_loss_pct': 5.0,
    }
    
    def create_env_fn():
        return EpisodicFairnessEnv(
            full_dataset_df=df.copy(), 
            static_feature_names_list=STATIC_FEATURES,
            driftable_feature_names_list=DRIFTABLE_FEATURES,
            ordered_feature_names_list=ALL_FEATURE_NAMES_ORDERED,
            target_column_name_for_proxy_training=TARGET_COLUMN_PROXY,
            proxy_model=proxy_model,
            max_episode_steps=10, 
            drift_config=drift_settings
        )
    

    train_env = create_env_fn() 
    train_env.reset(seed=SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FairDQN(
        "MlpPolicy",
        train_env,
        all_dataset_states=states_np_array,
        fair_pairs_list=fair_pairs,
        lambda_fair=10.0, 
        batch_size_pairs=128, 
        weighted_frac=0.5,   
        learning_rate=1e-3, 
        buffer_size=100_000, 
        learning_starts=5000,
        batch_size=256, 
        train_freq=1, 
        gradient_steps=1, 
        target_update_interval=1000, 
        exploration_fraction=0.1, 
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        tau=0.005, 
        gamma=0.99,
        seed=SEED,
        verbose=0,
        device=device 
    )

    callback = FairnessEvalCallback(
        initial_undrifted_states=states_np_array,
        fair_pairs_list=fair_pairs,
        true_labels_initial=true_labels_for_initial_states,
        sensitive_feature_name=SENSITIVE_ATTRIBUTE_NAME,
        all_feature_names_ordered_list=ALL_FEATURE_NAMES_ORDERED,
        driftable_feature_names_list=DRIFTABLE_FEATURES,
        eval_env_creator=create_env_fn,
        eval_freq=500, 
        n_eval_episodes_reward=50,
        n_eval_episodes_group_fairness=100, 
        sigma_final=sigma,
        k_final=k,
        batch_size_pairs=128,
        verbose=1
    )

    print("Starting training...")
    model.learn(total_timesteps=25_000, callback=callback) 
    
    print("\nTraining finished.")

    metrics_df = callback.get_metrics_log_df()
    metrics_df = metrics_df[metrics_df["step"] > model.learning_starts].reset_index(drop=True)
    if not metrics_df.empty:
        #Set options to display all rows and columns
        pd.set_option('display.max_rows', None)  # Show all rows
        pd.set_option('display.max_columns', None) # Show all columns
        pd.set_option('display.width', None)       # Adjust width to console
        pd.set_option('display.max_colwidth', None) # Show full content of each column
        print("\n--- Evaluation Metrics Over Time ---")
        print(metrics_df)
        metrics_df.to_csv("metrics.csv")
        
        # Adjust figure size and subplot layout for an additional plot
        plt.figure(figsize=(24, 12)) 
        
        plt.subplot(2, 5, 1) 
        plt.plot(metrics_df["step"], metrics_df["mean_reward_vs_Yproxy"], label="Mean Reward (vs Y_proxy)")
        plt.xlabel("Timestep")
        plt.ylabel("Reward")
        plt.legend()
        plt.title("Mean Reward")

        plt.subplot(2, 5, 2)
        plt.plot(metrics_df["step"], metrics_df["avg_lipschitz_penalty_train"], label="Avg Lipschitz Penalty (Initial States)")
        plt.plot(metrics_df["step"], metrics_df["lipschitz_on_final"], label="Avg Lipschitz Penalty (Drifted States)")
        plt.xlabel("Timestep")
        plt.ylabel("Lipschitz Penalty")
        plt.legend()
        plt.title("Average Lipschitz Penalty (Initial vs Drifted States)")

        plt.subplot(2, 5, 3)
        plt.plot(metrics_df["step"], metrics_df["cfd_on_initial"], label="CFD (Initial States)")
        plt.plot(metrics_df["step"], metrics_df["cfd_on_final"], label="CFD (Final States)")
        plt.xlabel("Timestep")
        plt.ylabel("Fairness Metric Value")
        plt.legend()
        plt.title("CFD (Initial vs. Drifted States)")

        plt.subplot(2, 5, 4)
        plt.plot(metrics_df["step"], metrics_df["consistency_on_initial"], label="Consistency (Initial States)")
        plt.plot(metrics_df["step"], metrics_df["consistency_on_final"], label="Consistency (Finall States)")
        plt.xlabel("Timestep")
        plt.ylabel("Fairness Metric Value")
        plt.legend()
        plt.title("Consistency (Initial vs. Drifted States)")

        plt.subplot(2, 5, 5)
        plt.plot(metrics_df["step"], metrics_df["dp_gap_episodic"], label="DP Gap (Episodic vs Y_proxy)")
        plt.xlabel("Timestep")
        plt.ylabel("DP Gap")
        plt.legend()
        plt.title("Demographic Parity Gap (Episodic)")

        plt.subplot(2, 5, 6)
        plt.plot(metrics_df["step"], metrics_df["eo_tpr_gap_episodic"], label="EO TPR Gap (Episodic vs Y_proxy)")
        plt.plot(metrics_df["step"], metrics_df["eo_fpr_gap_episodic"], label="EO FPR Gap (Episodic vs Y_proxy)")
        plt.xlabel("Timestep")
        plt.ylabel("EO Gap")
        plt.legend()
        plt.title("Equalized Odds Gaps (Episodic)")
        
        plt.subplot(2, 5, 7)
        plt.plot(metrics_df["step"], metrics_df["tpr_group0_ep"], label="TPR Group 0 (Episodic)")
        plt.plot(metrics_df["step"], metrics_df["tpr_group1_ep"], label="TPR Group 1 (Episodic)")
        plt.xlabel("Timestep")
        plt.ylabel("TPR")
        plt.legend()
        plt.title("True Positive Rates by Group (Episodic)")

        plt.subplot(2, 5, 8)
        plt.plot(metrics_df["step"], metrics_df["fpr_group0_ep"], label="FPR Group 0 (Episodic)")
        plt.plot(metrics_df["step"], metrics_df["fpr_group1_ep"], label="FPR Group 1 (Episodic)")
        plt.xlabel("Timestep")
        plt.ylabel("FPR")
        plt.legend()
        plt.title("False Positive Rates by Group (Episodic)")


        # Plot % drift of capital.gain by group
        plt.subplot(2, 5, 9)
        plt.plot(metrics_df["step"], metrics_df["drift_pct_capital_gain_grp0"], label="Group 0")
        plt.plot(metrics_df["step"], metrics_df["drift_pct_capital_gain_grp1"], label="Group 1")
        plt.xlabel("Timestep"); plt.ylabel("% Drift"); plt.legend()
        plt.title("Average % Drift: capital.gain")

        # Plot % drift of capital.loss by group
        plt.subplot(2, 5, 10)
        plt.plot(metrics_df["step"], metrics_df["drift_pct_capital_loss_grp0"], label="Group 0")
        plt.plot(metrics_df["step"], metrics_df["drift_pct_capital_loss_grp1"], label="Group 1")
        plt.xlabel("Timestep"); plt.ylabel("% Drift"); plt.legend()
        plt.title("Average % Drift: capital.loss")

        plt.tight_layout()
        plt.savefig("fairness_rl_episodic_training_curves_ext.png") 
        print("\nPlots saved to fairness_rl_episodic_training_curves_ext.png")
        plt.show()
    else:
        print("No metrics were logged by the callback. Check eval_freq and total_timesteps.")

    train_env.close()