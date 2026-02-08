import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="ML Assignment 2", layout="wide")

st.title("ML Classification Models – Assignment 2")

model_dict = {
    "Logistic Regression": "logistic.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

uploaded_file = st.file_uploader(
    "Upload CSV Test Dataset",
    type=["csv"]
)

selected_model = st.selectbox(
    "Select Model",
    list(model_dict.keys())
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    X = df.drop("target", axis=1)
    y = df["target"]

    model = joblib.load(f"model/saved_models/{model_dict[selected_model]}")

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", round((y_pred == y).mean(), 3))
    col2.metric("Precision", round(classification_report(y, y_pred, output_dict=True)["1"]["precision"], 3))
    col3.metric("Recall", round(classification_report(y, y_pred, output_dict=True)["1"]["recall"], 3))

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    st.subheader("📄 Classification Report")
    st.text(classification_report(y, y_pred))

