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

st.title("Feature Engineering")
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
# One-Hot Encoding
# -----------------------------------
    st.markdown("### One-Hot Encoding")

# st.code("""
# categorical_cols = [
#     'Device_Used',
#     'Payment_Method',
#     'Location',
#     'Transaction_Type'
# ]

# df2 = pd.get_dummies(
#     df2,
#     columns=categorical_cols,
#     drop_first=True
# )
# """, language="python")


# st.markdown("### Feature Encoding")

st.write(
    "Device_Used: One-Hot Encoding. Others: Frequency Encoding"
)

# categorical_cols = [
#     'Device_Used',
#     'Payment_Method',
#     'Location',
#     'Transaction_Type'
# ]

# df2 = pd.get_dummies(
#     df2,
#     columns=categorical_cols,
#     drop_first=True
# )
   # ----------------------------
    # 1. ONE-HOT: Device_Used
    # ----------------------------
df2 = pd.get_dummies(df2, columns=['Device_Used'], drop_first=True)

 # ----------------------------
    # 2. FREQUENCY ENCODING (others)
    # ----------------------------
freq_cols = ['Payment_Method', 'Location', 'Transaction_Type']

for col in freq_cols:
    freq_map = df2[col].value_counts(normalize=True)
    df2[col] = df2[col].map(freq_map)

st.subheader("Encoded Data")
st.write(df2.head())

# STORE it in session_state
st.session_state["df2"] = df2
st.write("Data encoded and saved!")

#st.dataframe(df2.head())