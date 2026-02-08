import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="ML Classification Model Explorer",
    page_icon="🤖",
    layout="wide"
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("⚙️ Control Panel")

# --------------------------------------------------
# Download Sample Dataset
# --------------------------------------------------
st.sidebar.markdown("### 📥 Sample Dataset")

with open("data/sample_dataset.csv", "rb") as file:
    st.sidebar.download_button(
        label="⬇️ Download Sample Test Dataset",
        data=file,
        file_name="sample_test_dataset.csv",
        mime="text/csv"
    )

st.sidebar.markdown(
    "<small>This dataset matches the trained model features.</small>",
    unsafe_allow_html=True
)

# --------------------------------------------------
# Upload Dataset
# --------------------------------------------------
st.sidebar.markdown("Upload dataset and select model")

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Test CSV",
    type=["csv"]
)

model_dict = {
    "Logistic Regression": "logistic.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbors": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest.pkl",
    "XGBoost (Ensemble)": "xgboost.pkl"
}

# --------------------------------------------------
# Target Column Selection
# --------------------------------------------------
st.sidebar.markdown("### 🎯 Target Variable")

target_col = st.sidebar.selectbox(
  "Select Target Column",
   df.columns
)
selected_model = st.sidebar.selectbox(
    "🧠 Choose Classification Model",
    list(model_dict.keys())
)

st.sidebar.markdown("---")

# --------------------------------------------------
# Main Header
# --------------------------------------------------
st.markdown(
    """
    <h1 style='text-align: center;'>📊 ML Classification Model Explorer</h1>
    <p style='text-align: center; color: grey;'>
    End-to-End ML Deployment using Streamlit
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# --------------------------------------------------
# Main Logic
# --------------------------------------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head(), width=True)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    model_path = f"model/saved_models/{model_dict[selected_model]}"
    model = joblib.load(model_path)

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------
    st.markdown("## 📈 Model Performance Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("✅ Accuracy", f"{accuracy_score(y, y_pred):.3f}")
    col2.metric("🎯 Precision", f"{precision_score(y, y_pred):.3f}")
    col3.metric("🔁 Recall", f"{recall_score(y, y_pred):.3f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("📊 F1 Score", f"{f1_score(y, y_pred):.3f}")
    col5.metric("📈 AUC", f"{roc_auc_score(y, y_prob):.3f}")
    col6.metric("🧮 MCC", f"{matthews_corrcoef(y, y_pred):.3f}")

    st.markdown("---")

    # --------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------
    st.markdown("## 🧩 Confusion Matrix")

    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    st.pyplot(fig)

    # --------------------------------------------------
    # Classification Report
    # --------------------------------------------------
    st.markdown("## 📄 Detailed Classification Report")
    report = classification_report(y, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, width=True)

else:
    st.info(
        "👈 Upload a test dataset from the sidebar to begin.\n\n"
        "This app demonstrates multiple ML classification models "
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown(
    """
    <hr>
    <p style='text-align:center; color: grey;'>
    Built with ❤️ using Streamlit 
    </p>
    """,
    unsafe_allow_html=True
)
