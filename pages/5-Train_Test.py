import streamlit as st
import pandas as pd
#import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

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

# -----------------------------------
# LOAD DATA
# -----------------------------------

if 'df2' in st.session_state:
    df2 = st.session_state['df2']

    st.write(df2.head())
else:
    st.error("DataFrame not found. Please run EDA first.")

    #st.write(df2.head())


# -----------------------------------
# Train Test
# -----------------------------------
    st.markdown("### Features and Target Split")

st.code("""
X = df2.drop(['Fraudulent'], axis=1)
y = df2['Fraudulent']
""", language="python")

# Execute the code
X = df2.drop(['Fraudulent'], axis=1)
y = df2['Fraudulent']

# Display results
st.write("### Features (X)")
st.dataframe(X.head())

st.write("### Target (y)")
st.dataframe(y.head())

st.write(f"Number of Features: {X.shape[1]}")
st.write(f"Number of Records: {X.shape[0]}")

# -----------------------------------
# Train Test
# -----------------------------------

from sklearn.model_selection import train_test_split

st.markdown("### Train-Test Split")

st.code("""
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Keeps fraud ratio balanced in train/test sets.
)
""", language="python")

# Execute the code
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# Display results
st.write("### Dataset Shapes")

col1, col2 = st.columns(2)

with col1:
    st.write("Training Set")
    st.write(f"X_train: {X_train.shape}")
    st.write(f"y_train: {y_train.shape}")

with col2:
    st.write("Testing Set")
    st.write(f"X_test: {X_test.shape}")
    st.write(f"y_test: {y_test.shape}")

# Show class distribution
st.write("### Class Distribution")

dist_df = {
    "Dataset": ["Train", "Test"],
    "Non-Fraud": [
        (y_train == 0).sum(),
        (y_test == 0).sum()
    ],
    "Fraud": [
        (y_train == 1).sum(),
        (y_test == 1).sum()
    ]
}

st.write("Machine Learning Fraud Detection System")
st.write(df2.head())

st.dataframe(dist_df)

st.session_state["X_train"] = X_train
st.session_state["X_test"] = X_test
st.session_state["y_train"] = y_train
st.session_state["y_test"] = y_test


# STORE it in session_state
st.session_state["df2"] = df2