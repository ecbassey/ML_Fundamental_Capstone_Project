import streamlit as st
import pandas as pd
#import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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

st.title("Explore the Data (EDA)")

st.write("Machine Learning Fraud Detection System")

# -----------------------------------
# LOAD DATA
# -----------------------------------

df = pd.read_csv('Fraud Detection Dataset.csv')

st.subheader("Dataset Preview")

st.dataframe(df.head())


# -----------------------------------
# Info
# -----------------------------------
st.write(df.isnull().sum())


# -----------------------------------
# Check Fraud Distribution
# -----------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.write("Check Fraud Distribution")
st.write(df['Fraudulent'].value_counts(normalize=True) * 100)


fig, ax = plt.subplots(figsize=(6, 4))
plot = sns.countplot(x='Fraudulent', data=df, ax=ax)
for container in plot.containers:
    ax.bar_label(container)
ax.set_title("Fraudulent Transaction Distribution")
st.pyplot(fig)



fig, ax = plt.subplots(figsize=(6, 4))
bars = df.groupby('Fraudulent')['Transaction_Amount'].mean().plot(
    kind='bar',
    ax=ax
)
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f')
ax.set_ylabel('Average Transaction Amount')
ax.set_xlabel('Fraudulent')
ax.set_title('Average Transaction Amount by Fraud Status')
st.pyplot(fig)

# -----------------------------------
# Drop Columns
# -----------------------------------
df = df.drop(columns=['Transaction_ID', 'User_ID'])
st.success("Transaction_ID and User_ID columns have been removed.")


# -----------------------------------
# Plot histograms
# -----------------------------------
# # Select numerical columns
# num_cols = df.select_dtypes(include=['int64', 'float64']).columns
# # Generate histograms
# axes = df[num_cols].hist(
#     figsize=(15, 12),
#     bins=30,
#     edgecolor="black"
# )
# plt.suptitle("Distribution of Numerical Columns", fontsize=16)
# plt.tight_layout()
# st.pyplot(plt.gcf())

# -----------------------------------
# Plot histograms  2
# -----------------------------------
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

fig, axes = plt.subplots(len(num_cols), 1, figsize=(10, 4 * len(num_cols)))

# if only 1 column, make it iterable
if len(num_cols) == 1:
    axes = [axes]

for ax, col in zip(axes, num_cols):
    sns.histplot(df[col], bins=30, kde=True, ax=ax)
    ax.set_title(f"Distribution of {col}")

plt.tight_layout()
st.pyplot(fig)



# ----- over 5000
count_over_5000 = (df["Transaction_Amount"] > 5000).sum()
total = len(df)
percent = (count_over_5000 / total) * 100

st.metric("Transactions > 5000", count_over_5000)
st.write(f"{percent:.2f}% of all transactions")

# ---- chart
# filter rows where amount > 5000
df_over_5000 = df[df["Transaction_Amount"] > 5000]

# percentage breakdown of Fraudulent column
percentages = df_over_5000["Fraudulent"].value_counts(normalize=True) * 100

st.write("Fraud distribution for transactions over 5000:")
st.dataframe(percentages)

st.write(" ")
st.write(" ")


#------------------------------
# missingness
# -------------------------

fig, ax = plt.subplots(figsize=(10, 6))
msno.matrix(df, ax=ax)
# Smaller fonts
ax.tick_params(axis='both', labelsize=8)
ax.set_title("Missing Values Matrix", fontsize=10)
st.pyplot(fig)

#------------------------------
#heat map
# -------------------------


fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    fmt=".3f",
    cmap="coolwarm",
    ax=ax
)
ax.set_title("Correlation Heatmap")
st.pyplot(fig)

st.write("No multicollinearity or redundant features.")
st.write(" ")
st.write(" ")

#-------------------------
y = df["Fraudulent"]

X = (
    df.drop(columns=["Fraudulent"])
      .select_dtypes(include=["int64", "float64"])
      .fillna(df.median(numeric_only=True))
)

X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(
    X_pca[y == 0, 0],
    X_pca[y == 0, 1],
    alpha=0.5,
    label="Non-Fraud"
)
ax.scatter(
    X_pca[y == 1, 0],
    X_pca[y == 1, 1],
    alpha=0.5,
    label="Fraud"
)

ax.set_title("Linear separability of Fraud Classes")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend()

st.pyplot(fig)

st.write("Not linearly separable.")
st.write(" ")
st.write(" ")

#-------------------------
st.dataframe(df.head())

# session
st.session_state['df'] = df