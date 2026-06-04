import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import Pipeline as make_pipeline_imb
from imblearn.pipeline import make_pipeline as make_pipeline_imb
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
if "preprocessor" not in st.session_state:
    st.error("Preprocessor not found. Please run the preprocessing page first.")
    st.stop()

preprocessor = st.session_state["preprocessor"]

#st.write("Preprocessor loaded successfully")
#st.write(preprocessor)

X_train = st.session_state["X_train"]
y_train = st.session_state["y_train"]
X_test = st.session_state["X_test"]
y_test = st.session_state["y_test"]

# -----------------------------------
# PAGE TITLE
# -----------------------------------

st.title("Train Multiple Models")
st.write("Machine Learning Fraud Detection System")

X_train = st.session_state["X_train"]
y_train = st.session_state["y_train"]

# -----------------------------------
# LOAD DATA
# -----------------------------------

if 'df2' in st.session_state:
    df2 = st.session_state['df2']

    st.write(df2.head())
else:
    st.error("DataFrame not found. Please run EDA first.")


# -----------------------------------
# Train Multiple Models
# -----------------------------------

st.markdown("### Model Training Pipeline")

st.code("""
models = {
    'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced'),
    'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
    'Support Vector Machine': SVC(kernel='rbf', class_weight='balanced',  probability=True, random_state=42)

      
""", language="python")

# -----------------------------
# ACTUAL STREAMLIT EXECUTION
# -----------------------------

trained_models = {}

models = {
    'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced'),
    'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
    'Support Vector Machine': SVC(kernel='rbf', class_weight='balanced',  probability=True, random_state=42),

    # 'K-Nearest Neighbors': make_pipeline_imb(
    #     SMOTE(random_state=42),
    #     KNeighborsClassifier()
    # ),

    # 'Support Vector Machine': make_pipeline_imb(
    #     SMOTE(random_state=42),
    #     SVC(probability=True, random_state=42)
    # ),

    # 'XGBoost': XGBClassifier(
    #     use_label_encoder=False,
    #     eval_metric='logloss',
    #     scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
    #     random_state=42
    # ),

    # 'Gradient Boosting': make_pipeline_imb(
    #     SMOTE(random_state=42),
    #     GradientBoostingClassifier(random_state=42)
    # )
}

progress = st.progress(0)
status_text = st.empty()

total_models = len(models)

for i, (name, model) in enumerate(models.items(), 1):

    status_text.text(f"Training {name}...")

    # Apply preprocessing
    if "SMOTE" in str(model):
        clf = make_pipeline_imb(preprocessor, model)
    else:
        clf = make_pipeline(preprocessor, model)

    clf.fit(X_train, y_train)

    trained_models[name] = clf

    progress.progress(i / total_models)

st.success("All models trained successfully!")

st.write("### Trained Models")
st.write(list(trained_models.keys()))


st.session_state["trained_models"] = trained_models
st.session_state["X_train"] = X_train
st.session_state["X_test"] = X_test
st.session_state["y_train"] = y_train
st.session_state["y_test"] = y_test