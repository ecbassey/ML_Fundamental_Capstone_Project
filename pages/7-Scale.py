import streamlit as st
import pandas as pd
#import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from imblearn.over_sampling import SMOTE

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

st.title("Scale Numerical Features")
st.write("Machine Learning Fraud Detection System")

# -----------------------------------
# LOAD DATA
# -----------------------------------

if 'df2' in st.session_state:
    df2 = st.session_state['df2']

    st.write(df2.head())
else:
    st.error("DataFrame not found. Please run EDA first.")

X_train = st.session_state["X_train"]
y_train = st.session_state["y_train"]
X_test = st.session_state["X_test"]
y_test = st.session_state["y_test"]


# -----------------------------------
# SCALE
# -----------------------------------

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer

st.markdown("### Feature Scaling")

st.code("""

scaler_standard = StandardScaler()
scaler_robust = RobustScaler()


robust_cols = ['Transaction_Amount']

standard_cols = ['Previous_Fraudulent_Transactions', 'Number_of_Transactions_Last_24H',
 'Time_of_Transaction', 'Account_Age']


""", language="python")

# ---- ACTUAL EXECUTION ----
scaler_robust = RobustScaler()
#scaler_minmax = MinMaxScaler()
scaler_standard = StandardScaler()

robust_cols = [
    'Transaction_Amount',
    
]

standard_cols = [
    'Previous_Fraudulent_Transactions',
    'Number_of_Transactions_Last_24H',
    'Time_of_Transaction',
    'Account_Age'
]

preprocessor = ColumnTransformer(
    transformers=[
        ('robust', scaler_robust, robust_cols),
        ('standard', scaler_standard, standard_cols)
    ]
)

st.write("### Preprocessor Created")
st.write(preprocessor)

# STORE it in session_state
st.session_state["df2"] = df2

# SAVE to session state
st.session_state["preprocessor"] = preprocessor

st.session_state["X_train"] = X_train
st.session_state["X_test"] = X_test
st.session_state["y_train"] = y_train
st.session_state["y_test"] = y_test
