# ind_fair_MDP

## Dataset

This project uses the **ACSIncome** dataset from the "Folktables" package, which is a predictable version of the US Census data derived from the American Community Survey (ACS). It was introduced by Ding et al. (2021) to provide a higher-quality, more realistic dataset for fairness research, addressing known issues in the original UCI Adult Income dataset.

For more details on the dataset and its creation, please refer to the original paper:

> Frances Ding, Moritz Hardt, John Miller, and Ludwig Schmidt. (2021). **Retiring Adult: New Datasets for Fair Machine Learning**. In *Advances in Neural Information Processing Systems (NeurIPS) 35*.
>
> **[Link to Paper on arXiv](https://arxiv.org/abs/2108.04884)**

## File Structure

This project is organized into several key Python scripts and their corresponding configuration files.

- `clean_adult_dataset.py`: A preprocessing script that takes the raw adult.csv dataset, performs one-hot encoding on categorical features, scales numerical values, and saves the result as cleaned_adult.csv. This is the first step in the pipeline.
- `one_run.py`: Contains the core logic for the RL environment and agent. This includes the `EpisodicFairnessEnv`, the custom `FairDQN`agent, and the `FairnessEvalCallback`used for detailed evaluation during training. This script is designed to be controlled by `config.yaml`.
    - `config.yaml`: Controls `one_run.py`. It allows you to configure all aspects of a single experiment—including file paths, dataset features, drift logic, and all model hyperparameters.
- `report_experiments.py`: An automated experiment runner designed to conduct all experiments for the research report. It systematically runs multiple trainings by looping through different parameter values (e.g., different fairness weights, drift intensities) and seeds. It is controlled by `experiments_config.yaml`.
    - `experiments_config.yaml`: Controls `report_experiments.py`. Here, you define the parameter sweeps you want to run. For example, you can specify a list of `lambda_fair` values to test, and the runner script will execute a full training cycle for each one.
- `report_plots.py`: A post-processing script used to generate summary plots and tables from the collected metrics of all the experiments run by report_experiments.py. This is useful for creating the final figures and tables for the report.

---

## How to run

### Step 1: Prepare the Dataset

First, you need to generate a cleaned dataset with no missing values, scaled numerical features, and one-hot encoded categorical features. 

For the report I have used the Financial Adult dataset. Make sure you have the original `adult.csv` file in your repository.  Run `python clean_adult_dataset.py`. This will create `cleaned_adult.csv`, which is used by all subsequent scripts.

### Step 2: Running a Single Experiment

This is useful for debugging, testing a new feature, or running a one-off configuration.

1. **Configure your run**: Open `config.yaml` and modify the parameters as needed. You can change the `drift_logic`, `fair_dqn_params`, file paths, etc. Detailed comments are added for all parameters for clarification
2. **Execute the script**: `python one_run.py`. The script will train the agent according to `config.yaml` and save the resulting metrics CSV and plot image to the specified output folders.

### Step 3: Running a Full Experimental Setup

This is the main workflow for generating the data for a report. This script will run many training loops automatically.

1. **Define your experiments**: Open `experiments_config.yaml`. In the `experiment_sweeps` section, define the lists of parameter values you want to test.
2. **Execute the experiment runner**: `python report_experiments.py`. 
 It will systematically work through every combination of parameters and seeds defined in the config, saving a separate CSV and plot file for each individual run in the `metrics` and `plots` directories. This process will take a long time.

### Step 4: Generating Summary Plots

This step is specific for the Report. It generates the visualizations and summary metrics to compare the different setups. **Execute the plotting script**: `python report_plots.py`.

---

## Configuration

The use of config `.yaml` files is essential for customization.

- **For a single run (`config.yaml`)**: You can control everything from the features used (`dataset_columns`) to the exact physics of the environment (`drift_logic`) to the agent's learning rate (`fair_dqn_params`).
- **For a full experimental setup (`experiments_config.yaml`)**: The `experiment_sweeps` section is the main part, where you define the parameter values.
