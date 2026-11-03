
import os
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():

    model = xgb.XGBClassifier()

    model.load_model(
        str(MODEL_DIR / "xgboost_fraud_model.json")
    )

    preprocessor = joblib.load(
        MODEL_DIR / "preprocessor.pkl"
    )

    input_columns = joblib.load(
        MODEL_DIR / "input_columns.pkl"
    )

    explainer = shap.TreeExplainer(model)

    return model, preprocessor, input_columns, explainer


model, preprocessor, input_columns, explainer = load_models()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("💳 Credit Card Fraud Detection")

st.write(
    "An ML-powered application to identify potentially "
    "fraudulent credit card transactions."
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    ["Dashboard", "Fraud Detection", "About Project"]
)


# ==================================================
# DASHBOARD
# ==================================================

if page == "Dashboard":

    st.header("📊 Fraud Detection Dashboard")

    st.write(
        "Monitor model performance and analyze "
        "fraud detection results."
    )

    st.divider()

    # Key Performance Indicators

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Selected Model", "XGBoost")
    col2.metric("ROC-AUC", "0.8891")
    col3.metric("PR-AUC", "0.4462")
    col4.metric("Decision Threshold", "80%")

    st.divider()

    # Transaction Distribution

    st.subheader("Transaction Distribution")

    distribution = pd.DataFrame({
        "Transaction Type": [
            "Legitimate",
            "Fraudulent"
        ],
        "Transactions": [
            90783,
            3704
        ]
    })

    col1, col2 = st.columns(2)

    col1.metric("Legitimate Transactions", "90,783")
    col2.metric("Fraudulent Transactions", "3,704")

    st.bar_chart(
        distribution.set_index("Transaction Type"),
        horizontal=True
    )

    total_transactions = distribution["Transactions"].sum()

    fraud_percentage = (
        distribution.loc[
            distribution["Transaction Type"] == "Fraudulent",
            "Transactions"
        ].iloc[0] / total_transactions
    ) * 100

    st.metric(
        "Fraud Rate",
        f"{fraud_percentage:.2f}%"
    )

    st.caption(
        "Distribution based on the validation dataset."
    )

    st.divider()

    # Model Comparison

    st.subheader("Model Performance Comparison")

    performance = pd.DataFrame({
        "Model": [
            "XGBoost",
            "Random Forest",
            "Decision Tree",
            "Logistic Regression",
            "Gaussian Naive Bayes"
        ],
        "ROC-AUC": [
            0.889139,
            0.872755,
            0.828947,
            0.821169,
            0.717901
        ],
        "PR-AUC": [
            0.446242,
            0.411708,
            0.391774,
            0.200509,
            0.091224
        ]
    })

    st.dataframe(
        performance,
        width="stretch",
        hide_index=True
    )

    st.subheader("ROC-AUC Comparison")

    st.bar_chart(
        performance.set_index("Model")[["ROC-AUC"]],
        horizontal=True
    )

    st.subheader("Precision-Recall AUC Comparison")

    st.bar_chart(
        performance.set_index("Model")[["PR-AUC"]],
        horizontal=True
    )

    st.divider()

    # Confusion Matrix

    st.subheader("Confusion Matrix")

    st.write(
        "XGBoost validation results at the selected "
        "0.80 decision threshold."
    )

    confusion = pd.DataFrame(
        [
            [88925, 1858],
            [1763, 1941]
        ],
        index=["Actual Legitimate", "Actual Fraud"],
        columns=["Predicted Legitimate", "Predicted Fraud"]
    )

    st.dataframe(
        confusion,
        width="stretch"
    )

    st.divider()

    # Threshold Analysis

    st.subheader("Decision Threshold Analysis")

    threshold_file = (
        BASE_DIR / "outputs" / "threshold_results.csv"
    )

    if threshold_file.exists():

        threshold_data = pd.read_csv(threshold_file)

        st.dataframe(
            threshold_data,
            width="stretch",
            hide_index=True
        )

        selected_threshold = st.slider(
            "Select Decision Threshold",
            min_value=0.30,
            max_value=0.87,
            value=0.80,
            step=0.01
        )

        nearest_index = (
            threshold_data["Threshold"] - selected_threshold
        ).abs().idxmin()

        selected_row = threshold_data.loc[nearest_index]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Precision",
            f"{selected_row['Precision']:.2%}"
        )

        col2.metric(
            "Recall",
            f"{selected_row['Recall']:.2%}"
        )

        col3.metric(
            "F1-Score",
            f"{selected_row['F1-Score']:.2%}"
        )

        st.caption(
            f"Nearest evaluated threshold: "
            f"{selected_row['Threshold']:.4f}"
        )

        st.line_chart(
            threshold_data.set_index("Threshold")[
                ["Precision", "Recall", "F1-Score"]
            ]
        )

        best_row = threshold_data.loc[
            threshold_data["F1-Score"].idxmax()
        ]

        st.success(
            f"Best F1-Score: {best_row['F1-Score']:.2%} "
            f"at threshold {best_row['Threshold']:.4f}"
        )

    else:

        st.error(
            "threshold_results.csv not found in outputs folder."
        )

    st.divider()

    # Confusion Matrix Heatmap

    st.subheader("Confusion Matrix Heatmap")

    cm = np.array([
        [88925, 1858],
        [1763, 1941]
    ])

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt=",",
        cmap="Blues",
        xticklabels=[
            "Predicted Legitimate",
            "Predicted Fraud"
        ],
        yticklabels=[
            "Actual Legitimate",
            "Actual Fraud"
        ],
        linewidths=1,
        ax=ax
    )

    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    ax.set_title("XGBoost Confusion Matrix (Threshold = 0.80)")

    st.pyplot(fig)
    plt.close(fig)


# ==================================================
# FRAUD DETECTION
# ==================================================

elif page == "Fraud Detection":

    st.header("Transaction Analysis")

    st.info(
        "Enter the transaction details below to estimate "
        "whether it is potentially fraudulent."
    )

    with st.form("transaction_form"):

        transaction = {}

        cols = st.columns(2)

        for i, feature in enumerate(input_columns):

            with cols[i % 2]:

                transaction[feature] = st.number_input(
                    feature,
                    value=0.0,
                    format="%.6f"
                )

        submitted = st.form_submit_button(
            "Predict Transaction"
        )

    if submitted:

        input_df = pd.DataFrame(
            [transaction],
            columns=input_columns
        )

        try:

            # Preprocess transaction

            processed = preprocessor.transform(input_df)

            # Convert sparse matrix if necessary

            if hasattr(processed, "toarray"):
                processed = processed.toarray()

            # Model prediction

            probability = model.predict_proba(processed)[0][1]

            OPTIMAL_THRESHOLD = 0.80

            is_fraud = probability >= OPTIMAL_THRESHOLD

            st.divider()

            st.subheader("Prediction Result")

            st.metric(
                "Fraud Probability",
                f"{probability * 100:.2f}%"
            )

            st.progress(float(probability))

            if is_fraud:

                st.error(
                    "⚠️ Potential Fraudulent Transaction"
                )

            else:

                st.success(
                    "✅ Likely Legitimate Transaction"
                )

            st.caption(
                "Classification uses the selected 0.80 decision threshold."
            )

            # ------------------------------------------
            # SHAP EXPLAINABILITY
            # ------------------------------------------

            st.divider()

            st.subheader("🔍 Model Explainability")

            st.write(
                "The following features show which inputs "
                "contributed most to this model prediction."
            )

            shap_explanation = explainer(processed)

            shap_values = np.asarray(
                shap_explanation.values
            )

            # Handle binary classification output shapes

            if shap_values.ndim == 3:

                if shap_values.shape[-1] == 2:
                    feature_contributions = shap_values[0, :, 1]
                else:
                    feature_contributions = shap_values[0, :, 0]

            elif shap_values.ndim == 2:

                feature_contributions = shap_values[0]

            else:

                feature_contributions = shap_values.flatten()

            # Get transformed feature names

            try:

                feature_names = (
                    preprocessor.get_feature_names_out()
                )

            except Exception:

                feature_names = np.array(input_columns)

            # Match names to SHAP values

            if len(feature_names) != len(feature_contributions):

                feature_names = [
                    f"Feature {i + 1}"
                    for i in range(len(feature_contributions))
                ]

            shap_df = pd.DataFrame({
                "Feature": feature_names,
                "SHAP Value": feature_contributions
            })

            shap_df["Absolute Impact"] = (
                shap_df["SHAP Value"].abs()
            )

            shap_df = shap_df.sort_values(
                "Absolute Impact",
                ascending=False
            ).head(10)

            st.write(
                "**Top 10 features influencing this prediction**"
            )

            st.dataframe(
                shap_df[
                    ["Feature", "SHAP Value"]
                ],
                width="stretch",
                hide_index=True
            )

            # SHAP visualization

            fig, ax = plt.subplots(figsize=(9, 5))

            plot_data = shap_df.sort_values(
                "SHAP Value"
            )

            ax.barh(
                plot_data["Feature"].astype(str),
                plot_data["SHAP Value"]
            )

            ax.axvline(
                0,
                linewidth=1
            )

            ax.set_xlabel("SHAP Contribution")
            ax.set_ylabel("Feature")
            ax.set_title("Top 10 SHAP Feature Contributions")

            fig.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

            st.caption(
                "Positive SHAP values push the model output "
                "toward the fraud class; negative values push "
                "it away. These explain model contributions, "
                "not causal relationships."
            )

        except Exception as e:

            st.error(
                f"Prediction or explanation error: {e}"
            )


# ==================================================
# ABOUT PROJECT
# ==================================================

elif page == "About Project":

    st.header("About the Project")

    st.write("""
    This project uses Machine Learning to detect fraudulent
    credit card transactions.

    **Dataset:** IEEE-CIS Fraud Detection

    **Selected Algorithm:** XGBoost Classifier

    **Additional Technique:** SHAP Explainable AI

    **Evaluation Metrics:**
    - ROC-AUC
    - PR-AUC
    - Precision
    - Recall
    - F1-score

    **Project Features:**
    - Transaction fraud prediction
    - Model performance dashboard
    - Decision threshold analysis
    - Confusion matrix visualization
    - SHAP-based model explainability

    The project addresses the challenge of highly
    imbalanced transaction data.
    """)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Credit Card Fraud Detection | "
    "Machine Learning Capstone Project"
)
