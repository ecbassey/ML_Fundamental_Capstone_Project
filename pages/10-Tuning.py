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
#from xgboost import XGBClassifier
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

st.title("Tuning")
#st.write("Machine Learning Fraud Detection System")

X_train = st.session_state["X_train"]
y_train = st.session_state["y_train"]

# ------------------------------
# tuning
# ------LR --------------
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

st.subheader("Tune Logistic Regression (Random Search)")

if st.button("Run LR Random Search"):

    param_dist = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],
        'class_weight': [None, 'balanced']
    }

    random_search = RandomizedSearchCV(
        estimator=LogisticRegression(max_iter=1000),
        param_distributions=param_dist,
        n_iter=10,              # Number of combinations to test
        cv=5,
        scoring='f1',
        random_state=42,
        n_jobs=-1
    )

    with st.spinner("Running Random Search..."):
        random_search.fit(X_train, y_train)

    st.success("Search Complete!")

    st.write("### Best Parameters")
    st.json(random_search.best_params_)

    st.write("### Best F1 Score")
    st.write(round(random_search.best_score_, 4))

    st.session_state["lr_best_model"] = random_search.best_estimator_


# train
best_lr = random_search.best_estimator_
best_lr.fit(X_train, y_train)
y_pred = best_lr.predict(X_test)

# Report
from sklearn.metrics import classification_report
st.text(classification_report(y_test, y_pred))



    # ------SVC --------------
    # ------SVC --------------
    # ------SVC --------------

# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.preprocessing import StandardScaler

# st.subheader("Tune Support Vector Classifier (Random Search)")

# if st.button("Run SVC Random Search"):

#     # Scale data
#     scaler = StandardScaler()

#     X_train_scaled = scaler.fit_transform(X_train)

#     param_dist = {
#         'C': [0.01, 0.1, 1, 10, 100],
#         'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
#         'kernel': ['linear', 'rbf'],
#         'class_weight': [None, 'balanced']
#     }

#     random_search = RandomizedSearchCV(
#         estimator=SVC(),
#         param_distributions=param_dist,
#         n_iter=20,
#         cv=5,
#         scoring='f1',
#         random_state=42,
#         n_jobs=-1
#     )
#     with st.spinner("Running SVC Random Search..."):
#         random_search.fit(X_train_scaled, y_train)

#     st.success("Search Complete!")

#     st.write("### Best Parameters")
#     st.json(random_search.best_params_)

#     st.write("### Best Cross-Validation F1 Score")
#     st.write(round(random_search.best_score_, 4))

#     # Save for later use
#     st.session_state["svc_best_model"] = random_search.best_estimator_
#     st.session_state["svc_scaler"] = scaler

# # evaluate
# best_svc = st.session_state["svc_best_model"]
# scaler = st.session_state["svc_scaler"]

# X_test_scaled = scaler.transform(X_test)
# y_pred = best_svc.predict(X_test_scaled)
# st.text(classification_report(y_test, y_pred))

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

st.subheader("Model Comparison")

if st.button("Evaluate All Models"):

    # models should already be stored like:
    # models = {"Logistic Regression": best_lr, "SVC": best_svc, ...}

    results = []

    for name, model in trained_models.items():

        y_pred = model.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred)
        })

    results_df = pd.DataFrame(results)

    st.write("### Model Performance Table")
    st.dataframe(results_df)

    st.write("### Best Model (by F1 Score)")
    best_model_row = results_df.sort_values("F1 Score", ascending=False).iloc[0]

    st.success(f"Best Model: {best_model_row['Model']}")
    st.write(best_model_row)


# ------------------------------------------
# session
X_train = st.session_state["X_train"]
y_train = st.session_state["y_train"]
st.session_state["lr_best_model"] = best_lr