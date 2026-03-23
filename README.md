<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=Credit%20Card%20Fraud%20Detection&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=35&desc=XGBoost%20%2B%20SMOTE%20%2B%20SHAP%20for%20transaction%20fraud%20scoring&descAlignY=55&descSize=16" width="100%"/>

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/AUC--ROC-0.97-22c55e?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Recall-94%25-22c55e?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-f59e0b?style=for-the-badge"/>
</p>

<p>
  <a href="#overview">Overview</a> •
  <a href="#key-results">Key results</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#dataset">Dataset</a> •
  <a href="#methodology">Methodology</a>
</p>

</div>

---

## Overview

> **A fraud detection pipeline that identifies fraudulent credit card transactions using gradient boosting with SMOTE oversampling and SHAP-based explanations.**

Credit card fraud accounts for billions in losses annually, yet fraudulent transactions make up less than 2% of all activity. This extreme class imbalance makes standard classifiers unreliable. This project builds a detection system that trains four models (Logistic Regression, Random Forest, XGBoost, LightGBM) on synthetic transaction data, applies SMOTE to handle class imbalance, and uses SHAP to explain individual predictions. A threshold optimization module balances false positive costs against missed fraud losses.

```
Problem   →  Detecting rare fraud events in highly imbalanced transaction data
Solution  →  XGBoost with SMOTE oversampling, SHAP explanations, and cost-based threshold tuning
Impact    →  AUC 0.97, catches 94% of fraud with only 3% false positive rate
```

---

## Key results

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.97 |
| Recall (fraud caught) | 94% |
| False positive rate | 3% |
| PR-AUC | 0.82 |
| Best model | XGBoost |

---

## Architecture

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Synthetic data  │───▶│  SMOTE           │───▶│  Feature         │
│  generation      │    │  oversampling    │    │  scaling         │
└──────────────────┘    └──────────────────┘    └────────┬─────────┘
                                                         │
                          ┌──────────────────────────────┘
                          ▼
              ┌──────────────────────┐    ┌──────────────────────┐
              │  Model training      │───▶│  Threshold           │
              │  (4 classifiers)     │    │  optimization        │
              └──────────────────────┘    └──────────┬───────────┘
                                                     │
                          ┌──────────────────────────┘
                          ▼
              ┌──────────────────────┐    ┌──────────────────────┐
              │  SHAP                │───▶│  Fraud scoring       │
              │  explanations        │    │  dashboard           │
              └──────────────────────┘    └──────────────────────┘
```

<details>
<summary><b>Project structure</b></summary>

```
project_17_fraud_detection/
├── data/
│   ├── fraud_transactions.csv         # Transaction dataset
│   └── generate_data.py               # Synthetic data generator
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # Data generation and loading
│   └── model.py                       # Training, evaluation, SHAP
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb   # SMOTE, scaling, interactions
│   ├── 03_modeling.ipynb              # Model training and CV
│   └── 04_evaluation.ipynb            # ROC, SHAP, cost analysis
├── app.py                             # Streamlit dashboard
├── requirements.txt
└── README.md
```

</details>

---

## Quickstart

```bash
# Clone and navigate
git clone https://github.com/guydev42/calgary-data-portfolio.git
cd calgary-data-portfolio/project_17_fraud_detection

# Install dependencies
pip install -r requirements.txt

# Generate transaction data
python data/generate_data.py

# Launch dashboard
streamlit run app.py
```

---

## Dataset

| Property | Details |
|----------|---------|
| Source | Synthetic transaction data modeled on real-world fraud patterns |
| Transactions | 10,000 |
| Fraud rate | ~2% (200 fraudulent transactions) |
| Features | 10 (amount, time, distance, velocity, merchant category) |
| Target | is_fraud (binary) |

---

## Tech stack

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white"/>
  <img src="https://img.shields.io/badge/LightGBM-9558B2?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/SHAP-4B8BBE?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
</p>

---

## Methodology

<details>
<summary><b>Class imbalance handling</b></summary>

- SMOTE (Synthetic Minority Over-sampling Technique) to balance training data
- class_weight="balanced" for Logistic Regression and Random Forest
- scale_pos_weight for XGBoost, is_unbalance for LightGBM
</details>

<details>
<summary><b>Model training</b></summary>

- Four classifiers: Logistic Regression, Random Forest, XGBoost, LightGBM
- 5-fold StratifiedKFold cross-validation
- Metrics: AUC-ROC, precision, recall, F1, PR-AUC
</details>

<details>
<summary><b>SHAP explainability</b></summary>

- TreeExplainer for gradient boosting models
- Global feature importance via mean absolute SHAP values
- Waterfall plots for individual transaction explanations
</details>

<details>
<summary><b>Threshold optimization</b></summary>

- Business cost model: FN cost ($500 fraud loss) vs FP cost ($25 friction)
- Sweep thresholds from 0.05 to 0.95 to minimize total cost
- Achieves 94% recall at 3% false positive rate
</details>

---

## Acknowledgements

Built as part of the [Calgary Data Portfolio](https://guydev42.github.io/calgary-data-portfolio/).

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

**[Ola K.](https://github.com/guydev42)**
</div>
