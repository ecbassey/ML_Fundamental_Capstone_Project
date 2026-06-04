import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

## ---------- BG ----------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #61A4AD;
    }
    </style>
    """,
    unsafe_allow_html=True
)
#--------------------------

# -----------------------------------
# PAGE TITLE
# -----------------------------------

st.title("Train and Test")
st.write("Machine Learning Fraud Detection System")

X_train = st.session_state["X_train"]
y_train = st.session_state["y_train"]
X_test = st.session_state["X_test"]
y_test = st.session_state["y_test"]



# -----------------------------------
# LOAD DATA
# -----------------------------------

if 'df2' in st.session_state:
    df2 = st.session_state['df2']

    st.write(df2.head())
else:
    st.error("DataFrame not found. Please run EDA first.")


    # -----------------------------------
# st.markdown("### SMOTE Oversampling")

# st.code("""
# from imblearn.over_sampling import SMOTE

# smote = SMOTE(random_state=42)

# X_train_smote, y_train_smote = smote.fit_resample(
#     X_train,
#     y_train
# )
# """, language="python")

# # Apply SMOTE
# smote = SMOTE(random_state=42)

# X_train_smote, y_train_smote = smote.fit_resample(
#     X_train,
#     y_train
# )


# # Display results
# st.write("### Dataset Shapes")

# col1, col2 = st.columns(2)

# with col1:
#     st.write("Before SMOTE")
#     st.write(f"X_train: {X_train.shape}")
#     st.write(f"y_train: {y_train.shape}")

# with col2:
#     st.write("After SMOTE")
#     st.write(f"X_train_smote: {X_train_smote.shape}")
#     st.write(f"y_train_smote: {y_train_smote.shape}")

# # Show class distribution before and after
# st.write("### Class Distribution")

# before_fraud = (y_train == 1).sum()
# before_non_fraud = (y_train == 0).sum()

# after_fraud = (y_train_smote == 1).sum()
# after_non_fraud = (y_train_smote == 0).sum()

# distribution = {
#     "Dataset": ["Before SMOTE", "After SMOTE"],
#     "Non-Fraud": [before_non_fraud, after_non_fraud],
#     "Fraud": [before_fraud, after_fraud]
# }

# st.dataframe(distribution)

# # Show fraud percentages
# st.write("### Fraud Percentage")

# st.write(
#     f"Before SMOTE: {(before_fraud / len(y_train) * 100):.2f}% Fraud"
# )

# st.write(
#     f"After SMOTE: {(after_fraud / len(y_train_smote) * 100):.2f}% Fraud"
# )



st.title("Model Training (Logistic Regression)")
# model
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)
# # train
# if st.button("Train Model"):
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)

#     st.subheader("Results")
#     st.text(classification_report(y_test, y_pred))

#---------------------------------

# -----------------------------
# Model
# -----------------------------
model = RandomForestClassifier(
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)

# STORE it in session_state
st.session_state["df2"] = df2

