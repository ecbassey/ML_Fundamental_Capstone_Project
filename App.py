import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import base64

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)


## ---------- BG ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(
    -45deg,
        #8fa7b8,
        #9dab8d,
        #84bab7,
        #eddbd3
    );
    background-size: 400% 400%;
    animation: gradient 20s ease infinite;
}

@keyframes gradient {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
</style>
""", unsafe_allow_html=True)

#--------------------------

# -----------------------------------
# PAGE TITLE
# -----------------------------------

st.title("Credit Card  Fraud Detection")
st.write("Machine Learning Models for Fraudulent Transaction Detection")
# -----------------------------------


# -----------------------------------
st.markdown("## Problem Statement")

st.write("""
Detecting fraudulent financial transactions presents a challenging classification problem due to the significant imbalance between legitimate and fraudulent activities. The objective of this project is to develop and evaluate machine learning models that can accurately identify fraudulent transactions while maximizing fraud detection performance. Particular emphasis is placed on optimizing key evaluation metrics such as recall, precision, and F1-score to ensure effective detection of fraudulent activities while minimizing false alarms.
""")

st.markdown("## 📊 Dataset")

st.write("""This dataset captures 51,000+ transactions, each labeled as fraudulent or legitimate, based on real-world patterns.
""")


st.markdown("## 🔍 Insights and Solutions")

st.write("""
This project aims to uncover:
""")


st.markdown("""

- Which transactions are likely fraud (Fraud detection modeling)
- What patterns separate fraud from legitimate activity
- How can fraud be detected early (Anomaly detection)

""")