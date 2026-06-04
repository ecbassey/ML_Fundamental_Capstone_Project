import streamlit as st
import pandas as pd
#import joblib
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
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
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

st.write("Preprocessor loaded successfully")
#st.write(preprocessor)

trained_models = st.session_state.get("trained_models", None)
X_train = st.session_state.get("X_train", None)
X_test = st.session_state.get("X_test", None)
y_train = st.session_state.get("y_train", None)
y_test = st.session_state.get("y_test", None)

if trained_models is None:
    st.error("Models not found. Please train the models first.")
    st.stop()

# -----------------------------------
# PAGE TITLE
# -----------------------------------

st.title("Evaluate Models")
#st.write("Machine Learning Fraud Detection System")

X_train = st.session_state["X_train"]
y_train = st.session_state["y_train"]



# -----------------------------------

st.subheader("Model Evaluation Results")

results = {}

for name, model in trained_models.items():

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Training Metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, zero_division=0)
    train_recall = recall_score(y_train, y_train_pred, zero_division=0)
    train_f1 = f1_score(y_train, y_train_pred, zero_division=0)

    # Testing Metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)

     # Store Results
    results[name] = {
        "Train Accuracy": train_accuracy,
        "Train Precision": train_precision,
        "Train Recall": train_recall,
        "Train F1": train_f1,
        "Test Accuracy": test_accuracy,
        "Test Precision": test_precision,
        "Test Recall": test_recall,
        "Test F1": test_f1
    }

# Create DataFrame
results_df = pd.DataFrame(results).T

# Format to 4 decimal places
results_df = results_df.round(4)

# Display in Streamlit
st.dataframe(results_df, use_container_width=True)


# -----------------------------------
# Show as Metrics

selected_model = st.selectbox(
    "Select Model",
    results_df.index
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Test Accuracy",
              f"{results_df.loc[selected_model, 'Test Accuracy']:.3f}")

with col2:
    st.metric("Test Precision",
              f"{results_df.loc[selected_model, 'Test Precision']:.3f}")

with col3:
    st.metric("Test Recall",
              f"{results_df.loc[selected_model, 'Test Recall']:.3f}")

with col4:
    st.metric("Test F1",
              f"{results_df.loc[selected_model, 'Test F1']:.3f}")
    

#-----------------------
# confusion matrix
# Predictions
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Plot
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")

st.pyplot(fig)


#------------
st.session_state["trained_models"] = trained_models
st.session_state["X_train"] = X_train
st.session_state["X_test"] = X_test
st.session_state["y_train"] = y_train
st.session_state["y_test"] = y_test