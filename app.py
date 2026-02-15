import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef, 
                             confusion_matrix, classification_report)

# Import models from the local model/ directory
from model import (logistic_regression, decision_tree, knn, 
                   naive_bayes, random_forest, xgboost_model)

st.set_page_config(page_title="ML Assignment 2", layout="wide")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    df = pd.read_csv(url, header=None)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y

# --- DYNAMIC MODEL TRAINING ---
@st.cache_resource
def get_trained_models(_X_train, _y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(_X_train)
    
    # Initialize models from the imported custom scripts
    models = {
        "Logistic Regression": logistic_regression.get_model(),
        "Decision Tree": decision_tree.get_model(),
        "KNN": knn.get_model(),
        "Naive Bayes": naive_bayes.get_model(),
        "Random Forest": random_forest.get_model(),
        "XGBoost": xgboost_model.get_model()
    }
    
    for name, model in models.items():
        model.fit(X_train_scaled, _y_train)
        
    return models, scaler

# --- MAIN APP UI ---
st.title("Machine Learning Classification App")

# --- STUDENT DETAILS ADDED HERE ---
st.markdown("### ML Assignment 2")
st.markdown("**Manan Kumar Garg - 2025AA05493**")
st.markdown("---")

st.write("### Dataset: UCI Spambase (4,601 instances, 57 features)")

X, y = load_data()
X_train, X_test_default, y_train, y_test_default = train_test_split(X, y, test_size=0.2, random_state=42)
models, scaler = get_trained_models(X_train, y_train)

# --- SIDEBAR UI ---
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Custom Test CSV", type=["csv"])
selected_model_name = st.sidebar.selectbox("Select Model", list(models.keys()))

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file, header=None)
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]
    st.sidebar.success("Using uploaded test data.")
else:
    X_test = X_test_default
    y_test = y_test_default
    st.sidebar.info("Using default 20% test split.")
    
    # Utility to download a sample test file for the assignment's upload requirement
    test_export = X_test_default.copy()
    test_export['target'] = y_test_default
    csv = test_export.to_csv(index=False, header=False).encode('utf-8')
    st.sidebar.download_button("Download Sample Test CSV", data=csv, file_name="sample_test_data.csv", mime="text/csv")

X_test_scaled = scaler.transform(X_test)
selected_model = models[selected_model_name]
y_pred = selected_model.predict(X_test_scaled)
y_prob = selected_model.predict_proba(X_test_scaled)[:, 1]

# --- ALL MODELS COMPARISON TABLE ---
st.subheader("Comparison Table)")
results = []
for name, m in models.items():
    p = m.predict(X_test_scaled)
    prob = m.predict_proba(X_test_scaled)[:, 1]
    results.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, p),
        "AUC": roc_auc_score(y_test, prob),
        "Precision": precision_score(y_test, p, zero_division=0),
        "Recall": recall_score(y_test, p, zero_division=0),
        "F1": f1_score(y_test, p, zero_division=0),
        "MCC": matthews_corrcoef(y_test, p)
    })

results_df = pd.DataFrame(results).round(4)
st.dataframe(results_df, use_container_width=True)
st.markdown("---")

# --- SELECTED MODEL METRICS ---
st.subheader(f"Detailed Evaluation: {selected_model_name}")
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
col2.metric("AUC Score", f"{roc_auc_score(y_test, y_prob):.4f}")
col3.metric("F1 Score", f"{f1_score(y_test, y_pred):.4f}")

col4, col5, col6 = st.columns(3)
col4.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
col5.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
col6.metric("MCC Score", f"{matthews_corrcoef(y_test, y_pred):.4f}")

# --- CONFUSION MATRIX & REPORT ---
col_cm, col_cr = st.columns(2)
with col_cm:
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

with col_cr:
    st.write("### Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    # Fix the table error by dropping the 'accuracy' row before displaying
    report_df = pd.DataFrame(report).transpose().round(4)
    report_df = report_df.drop('accuracy', errors='ignore')
    st.dataframe(report_df)