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

if 'df' in st.session_state:
    df = st.session_state['df']

    #st.write(df.head())
else:
    st.error("DataFrame not found. Please run EDA first.")


df2 = df.copy()

# Create feature
st.write("Create 'High_Amount' feature from 'Transaction_Amount' where 'Transaction_Amount' is greater than Mean.")

st.write("**df2['High_Amount'] = (df2['Transaction_Amount'] > df2['Transaction_Amount'].mean()).astype(int)**")

df2['High_Amount'] = (
    df2['Transaction_Amount'] > df2['Transaction_Amount'].mean()
).astype(int)

# Display dataframe
st.write(df2.head())
#st.dataframe(df2)

# Show feature statistics
st.write("#### High Amount Distribution")
st.write(df2['High_Amount'].value_counts())


fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(
    df2.corr(numeric_only=True),
    annot=True,
    fmt=".3f",
    cmap="coolwarm",
    ax=ax
)
ax.set_title("Correlation Heatmap")
st.pyplot(fig)



st.session_state['df2'] = df2