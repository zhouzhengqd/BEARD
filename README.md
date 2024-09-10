<p align="center">
<img src="fig/logo.png" width="100%" class="center" alt="pipeline"/>
</p>

Welcome to the official PyTorch implementation of [BEARD](https://arxiv.org/). BEARD is an open-source benchmark specifically designed to evaluate and improve the **adversarial robustness** of Dataset Distillation (DD) methods.

Discover the **official leaderboard** here: **[BEARD Leaderboard](https://beard-leaderboard.github.io/)**


**❗Note❗: If you encounter any issues, please feel free to contact us!**

## 🚀 What's New?

- **September 2024**: The full BEARD codebase is now open-source! Access it here: [BEARD GitHub Repository](https://github.com/).
- **August 2024**: The first full release of the BEARD benchmark project.

## 🎯 Overview of BEARD

<p align="center">
<img src="fig/overview.png" width="100%" class="center" alt="pipeline"/>
</p>

BEARD addresses a critical gap in **Dataset Distillation (DD)** research by providing a systematic framework for evaluating adversarial robustness. While significant progress has been made in DD, deep learning models trained on distilled datasets remain vulnerable to adversarial attacks, posing risks in real-world applications.

### 🔥 Key Features:
- **Unified Benchmark**: Evaluate DD methods across multiple datasets and attack scenarios.
- **New Evaluation Metrics**: Includes the **Robustness Ratio (RR)**, **Attack Efficiency Ratio (AE)**, and **Comprehensive Robustness-Efficiency Index (CREI)**.
- **Open-Source Tools**: Easily integrate and evaluate the robustness of your DD methods with BEARD's extensible framework.

## 🛠 Getting Started
Follow the steps below to set up the environment and run the BEARD benchmark.

### Step 1: Clone the Repository
- Run the following command to download the Repo.
  ```
  git clone https://github.com/zhouzhengqd/BEARD.git
  ```
### Step 2: Download Dataset and Model Pools
- Download the [Dataset Pool](https://share.multcloud.link/share/a51b64d1-063c-4a5c-a7b2-667cf94da71a) and [Model Pool](https://share.multcloud.link/share/7dd850f1-b263-4f8b-9777-8e3134250187) and put them in the relevant positions.

### Step 3: Set Up the Conda Environment
- Run the following command to create a conda environment
    ```
    cd BEARD
    cd Code
    conda env create -f environment.yml
    conda activate beard
    ```
## 📁 Directory Structure
- BACON
    - Code
        - data
          - datasets
        - dataset_pool
        - model_pool
        - Files for BEARD
        - enviroment.yml
        - ...
        - ...
        - ...
## 🚦 Quick Evaluation Command
### Step 1: Download Dataset and Model Pools
- Ensure you have downloaded the dataset and model pools from the links provided above.
### Step 2: Modify Evaluation Configuration
- Adjust the evaluation configuration by editing the `evaluate_config.json` file based on your requirements.
### Step 3: Run the Evaluation Script
- Execute the evaluation to assess adversarial robustness:
  ```
    python evaluate_model.py --config ./evaluate_config.json
  ```
## ➕ Adding New Datasets and Models
### Step 1: Add Datasets
- Place the newly generated distilled datasets in the `dataset_pool` directory.
### Step 2: Modify Training Configuration
- Adjust the training configuration by editing the `train_config.json` file to specify the new datasets.
### Step 3: Run the Training Script
- Train the models on the new datasets:
  ```
    python train_model.py --config ./train_config.json
  ```
### Step 4: Evaluate the Models
- Once the models are trained, follow the evaluation steps outlined in the "Quick Evaluation Command" section to evaluate adversarial robustness.

## 🌐 Join the Community
If you're working on DD or adversarial robustness, we invite you to contribute to the BEARD benchmark, explore the leaderboard, and share your insights.

## 🙏 Acknowledgments

We would like to thank the contributors of the following projects that inspired and supported this work:

- [DC, DSA, DM](https://github.com/VICO-UoE/DatasetCondensation)
- [MTT](https://github.com/GeorgeCazenavette/mtt-distillation)
- [IDM](https://github.com/uitrbn/IDM)
- [BACON](https://github.com/zhouzhengqd/BACON)

