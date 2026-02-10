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
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="ML Classification Model for Breast Cancer Wisconsin (Diagnostic)",
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

selected_model = st.sidebar.selectbox(
    "🧠 Choose Classification Model",
    list(model_dict.keys())
)



# --------------------------------------------------
# Main Header
# --------------------------------------------------
st.markdown("""
<div style="text-align: center; padding: 1.5rem 0;">
    <h1 style="margin-bottom: 0.3rem;">
        📊 ML Classification Model Explorer
    </h1>
    <h3 style="margin-top: 0; font-weight: 400;">
        Breast Cancer Wisconsin (Diagnostic)
    </h3>
    <p style="color: grey; margin-top: 0.5rem;">
        End-to-End ML Deployment using Streamlit
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# --------------------------------------------------
# Main Logic
# --------------------------------------------------
if uploaded_file:
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Dataset Preview")
    st.markdown(
        f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns"
        )
    preview_df = df.head(10).reset_index(drop=True)
    st.write(preview_df)

    if df.empty:
        st.error("❌ Uploaded dataset is empty.")
        st.sidebar.markdown("---")
        st.stop()

    # -------------------------------
    # Missing data elements
    # --------------------------------
    st.markdown("## 🧪 Missing Values Check")
    threshold = 0.30

    missing_stats = pd.DataFrame({
        "Feature": df.columns,
        "Missing Count": df.isna().sum(),
        "Missing (%)": (df.isna().mean() * 100).round(2)
    })

    features_to_drop = missing_stats[
     missing_stats["Missing (%)"] > (threshold * 100)
    ]["Feature"].tolist()

    st.dataframe(missing_stats, use_container_width=True)

    if features_to_drop:
        st.warning(
            f"⚠️ Features dropped due to >30% missing values:\n"
            f"{features_to_drop}"
        )
        df = df.drop(columns=features_to_drop)
    else:
        st.success("✅ No features exceed the missing value threshold.")
   
    # --------------------------------------------------
    # Target Column Selection
    # --------------------------------------------------
    #st.sidebar.markdown("### 🎯 Target Variable")
    #target_col = st.sidebar.selectbox(
        #"Select Target Column",
        #options=["-- Select --"] + list(df.columns))
    target_col = 'diagnosis'
    # -------------------------------
    # Guard Clause
    # -------------------------------
    #if target_col == "-- Select --":
        #st.info("👈 Please select a target variable to continue.")
        #st.sidebar.markdown("---")
        #st.stop()
    run_model = st.sidebar.button("🚀 Run Model Evaluation")

    if not run_model:
        #st.info("👈 Select a target variable and click **Run Model Evaluation**.")
        st.sidebar.markdown("---")
        st.stop()
    st.sidebar.markdown("---")
   
    # -------------------------------
    # SAFE EXECUTION ZONE
    # -------------------------------
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # y is the selected target column
    if y.dtype == "object" or y.dtype.name == "category":
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        class_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

        st.info(
        f"🔁 Target labels encoded automatically:\n{class_mapping}"
        )
    else:
        y_encoded = y

    #re-assign the target data
    y = y_encoded
    model_path = f"model/saved_models/{model_dict[selected_model]}"
    model = joblib.load(model_path)

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------
    st.markdown("## 📈 Model Performance Metrics")

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("✅ Accuracy", f"{accuracy_score(y, y_pred):.3f}")
    col2.metric("📈 AUC", f"{roc_auc_score(y, y_prob):.3f}")
    col3.metric("🎯 Precision", f"{precision_score(y, y_pred):.3f}")
    col4.metric("🔁 Recall", f"{recall_score(y, y_pred):.3f}")
    col5.metric("📊 F1 Score", f"{f1_score(y, y_pred):.3f}")
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
    report_df = pd.DataFrame(report).transpose().round(3)
    report_df.index = report_df.index.astype(str)
    st.write(report_df)

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
