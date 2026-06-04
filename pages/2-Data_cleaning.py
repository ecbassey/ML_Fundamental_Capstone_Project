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

st.title("DATA CLEANING")

st.write("Machine Learning Fraud Detection System")

# -----------------------------------
# LOAD DATA
# -----------------------------------

if 'df' in st.session_state:
    df = st.session_state['df']

    st.write(df.head())
else:
    st.error("DataFrame not found. Please run EDA first.")

# -------
# check mean vs median
# --------------
mean_val = df["Transaction_Amount"].mean()
mean_under_5000 = df[df["Transaction_Amount"] < 5000]["Transaction_Amount"].mean()
median_val = df["Transaction_Amount"].median()


st.write("### Transaction_Amount Summary")

st.write(f"Mean: {mean_val:.2f}")
st.write(f"Mean Transaction Amount (< 5000): {mean_under_5000:.2f}")
st.write(f"Median: {median_val:.2f}")


# -----------------------------------
# Deal with the missing row.
# -----------------------------------
st.markdown("""
### Missing Value Treatment Applied

- **Transaction_Amount** → Median Imputation
- **Time_of_Transaction** → Mean Imputation
- **Device_Used** → 'Unknown Device'
- **Payment_Method** → Missing Indicator + 'Unknown'
- **Location** → Missing Indicator + 'Unknown'
""")

# Numerical columns
df['Transaction_Amount'] = df['Transaction_Amount'].fillna(
    df['Transaction_Amount'].median()
)

df['Time_of_Transaction'] = df['Time_of_Transaction'].fillna(
    df['Time_of_Transaction'].mean()
)

# Categorical columns with missing indicators
df['Device_Used'] = df['Device_Used'].fillna('Unknown Device')

#df['Payment_Method_Missing'] = df['Payment_Method'].isna().astype(int)
df['Payment_Method'] = df['Payment_Method'].fillna('Unknown')

#df['Location_Missing'] = df['Location'].isna().astype(int)
df['Location'] = df['Location'].fillna('Unknown')

# Save updated dataframe for other pages
st.session_state['df'] = df

st.success("Missing values have been handled successfully.")

# Display remaining missing values
st.subheader("Remaining Missing Values")
st.dataframe(
    df.isnull().sum().reset_index().rename(
        columns={"index": "Column", 0: "Missing Values"}
    )
)

# Preview updated dataset
st.subheader("Updated Dataset Preview")
st.dataframe(df.head())
