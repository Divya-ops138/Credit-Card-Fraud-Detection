# 💳 Credit Card Fraud Detection Using Machine Learning

An end-to-end machine learning project that identifies potentially fraudulent transactions using the IEEE-CIS Fraud Detection dataset.

##  Project Overview

Credit card fraud detection is a binary classification problem involving highly imbalanced transaction data. This project explores multiple machine learning algorithms, compares their performance, and uses XGBoost as the selected model for fraud prediction.

The project also includes an interactive Streamlit dashboard and SHAP-based explainability to understand model predictions.

##  Objectives

- Detect potentially fraudulent transactions.
- Handle class imbalance in transaction data.
- Compare multiple machine learning algorithms.
- Evaluate models using ROC-AUC and PR-AUC.
- Optimize the decision threshold.
- Explain predictions using SHAP.
- Deploy an interactive fraud detection application.

##  Dataset

**Dataset:** IEEE-CIS Fraud Detection  
**Source:** [Kaggle – IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection)

- Approximately 590,540 transaction records.
- Hundreds of original features.
- Binary target variable: `isFraud`.
- Highly imbalanced classification problem.

##  Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Random Forest
- Matplotlib and Seaborn
- Streamlit
- SHAP
- Joblib

##  Machine Learning Models

The following algorithms were evaluated:

1. XGBoost
2. Random Forest
3. Decision Tree
4. Logistic Regression
5. Gaussian Naive Bayes

## 📊 Model Performance

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| XGBoost | 0.8891 | 0.4462 |
| Random Forest | 0.8728 | 0.4117 |
| Decision Tree | 0.8289 | 0.3918 |
| Logistic Regression | 0.8212 | 0.2005 |
| Gaussian Naive Bayes | 0.7179 | 0.0912 |

**Selected Model: XGBoost**

##  Decision Threshold Analysis

The project evaluates multiple classification thresholds to understand the trade-off between precision and recall.

The selected operating threshold is **0.80**.

At this threshold, validation performance was:

- Precision: 51.09%
- Recall: 52.40%
- F1-Score: 51.74%

##  Model Explainability Using SHAP

SHAP (SHapley Additive exPlanations) is integrated to explain individual model predictions.

The application displays:
- Top 10 influential features.
- SHAP contribution values.
- Feature contribution visualization.

Positive SHAP values push the prediction toward fraud, while negative values push it away from fraud.

##  Streamlit Application

The interactive application includes:

- **Dashboard:** Model comparison, transaction distribution, evaluation metrics and threshold analysis.
- **Fraud Detection:** Enter transaction features and obtain a fraud probability.
- **Model Explainability:** Understand the features influencing each prediction.
- **About Project:** Project information and methodology.

##  Run Locally

Clone the repository:

```bash
git clone https://github.com/Divya-ops138/Credit-Card-Fraud-
