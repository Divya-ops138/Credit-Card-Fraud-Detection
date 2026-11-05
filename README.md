# Credit Card Fraud Detection Using Machine Learning

## Live Demo

[Try the Credit Card Fraud Detection App](https://credit-card-fraud-detection-5nwwhtdqpsf2kuxyvamfrw.streamlit.app/)

## About the Project

Credit card fraud is a major challenge in online transactions. Detecting fraudulent activity is difficult because fraudulent transactions make up only a small portion of the total transactions.

This project focuses on identifying potentially fraudulent credit card transactions using machine learning. I worked with transaction data, explored different classification algorithms, and compared their performance to find a model that can distinguish between legitimate and fraudulent transactions.

The main objective was to build a fraud detection system that considers the imbalance in the dataset and evaluates models using suitable performance metrics rather than relying only on accuracy.

## Dataset

The project uses the **IEEE-CIS Fraud Detection dataset** from Kaggle.

- Approximately 590,540 transaction records
- More than 400 original features
- Binary classification: fraudulent or legitimate transaction
- Highly imbalanced distribution of target classes

The dataset was explored and preprocessed before training the machine learning models.

## Machine Learning Models

I implemented and compared the following algorithms:

- Logistic Regression
- Gaussian Naive Bayes
- Decision Tree
- Random Forest
- XGBoost

After comparing their results, XGBoost was selected as the final model because it achieved the best performance among the models tested.

## Model Performance

The models were evaluated using ROC-AUC and PR-AUC scores.

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| XGBoost | 0.8891 | 0.4462 |
| Random Forest | 0.8728 | 0.4117 |
| Decision Tree | 0.8289 | 0.3918 |
| Logistic Regression | 0.8212 | 0.2005 |
| Gaussian Naive Bayes | 0.7179 | 0.0912 |

XGBoost achieved the highest ROC-AUC and PR-AUC scores in the model comparison.

## Application Features

The project includes a Streamlit application with the following features:

- **Fraud Detection:** Enter transaction details and obtain a fraud prediction with its estimated probability.
- **Model Comparison:** Compare the performance of the implemented classification algorithms.
- **Transaction Analysis:** View the distribution of legitimate and fraudulent transactions.
- **Performance Evaluation:** Explore ROC-AUC, PR-AUC, and confusion matrix visualizations.

## Technology Stack

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost
- **Visualization:** Matplotlib, Seaborn
- **Application:** Streamlit
- **Dataset:** Kaggle – IEEE-CIS Fraud Detection

## Model Files

The trained model and preprocessing components are saved and reused by the Streamlit application.

- `xgboost_fraud_model.json` – Trained XGBoost model
- `preprocessor.pkl` – Saved preprocessing pipeline
- `input_columns.pkl` – Feature information required by the model

The project does not use a separate SQL or NoSQL database. The dataset is processed using Pandas, while the trained model and preprocessing components are stored as files.

## Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── data/
│   └── raw/
│
├── models/
│
├── notebooks/
│
├── outputs/
│
├── src/
│   ├── app.py
│   └── test_model.py
│
├── requirements.txt
└── README.md
```

## What I Learned

Through this project, I gained practical experience in working with an imbalanced dataset and preparing transaction data for machine learning. I learned how to implement and compare multiple classification algorithms, evaluate their performance using different metrics, and select a suitable model.

I also gained experience in saving trained models, building an interactive application with Streamlit, and deploying a machine learning project for public access.
