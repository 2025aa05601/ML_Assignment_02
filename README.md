# ML Assignment 2 – Classification Models & Streamlit Deployment

## Problem Statement
To build a machine learning classifier that predicts whether a breast tumor is benign or malignant based on diagnostic features extracted from breast tissue images

## Dataset Description
The Breast Cancer Wisconsin (Diagnostic) dataset is a widely used benchmark dataset in machine learning and medical research. 
It is designed to predict whether a breast tumor is benign or malignant based on features computed from digitized images of fine needle aspirate (FNA) of breast masses
The Breast Cancer Wisconsin (Diagnostic) dataset contains 569 samples with 30 numerical features extracted from breast mass images and is used to classify tumors as benign or malignant

## Models Used
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors
- Naive Bayes
- Random Forest (Ensemble)
- XGBoost (Ensemble)

## Model Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|-----|---------|-----|----------|-------|----|----|
| Logistic Regression | 0.982 | 0.997 | 0.995 | 0.958 | 0.976 | 0.963 |
| Decision Tree | 0.982 | 0.980 | 0.981 | 0.972 | 0.976 | 0.962 |
| K-Nearest Neighbors | 0.974 | 0.995 | 0.990 | 0.939 | 0.964 | 0.944 |
| Naive Bayes | 0.938 | 0.989 | 0.936 | 0.896 | 0.916 | 0.868 |
| Random Forest (Ensemble) | 0.995 | 1.000 | 1.000 | 0.986 | 0.993 | 0.989 |
| XGBoost (Ensemble) | 0.995 | 1.000 | 1.000 | 0.986 | 0.993 | 0.989 |

## Observations

| ML Model Name | Observation about model performance |
|-----|-------------|
| Logistic Regression | 1. Very high AUC (0.997)<br/>2. Precision is extremely high (0.995)<br/>3. Slightly lower recall (0.958) ==> Some malignant cases may be missed |
| Decision Tree | 1. Same accuracy as Logistic (0.982)<br/>2. Balanced precision(0.981) and recall(0.972).<br/>3. Slightly lower AUC compared to Logistic. |
| K-Nearest Neighbors | 1. Good AUC (0.995)<br/>2. Recall slightly lower (0.939) ==> May miss some positive cases<br/>3. Sensitive to scaling and distance metric. |
| Naive Bayes | 1. Lowest accuracy (0.938)<br/>2. Lower MCC (0.868) ==> Less reliable classification.<br/>3. Assumes feature independence, which is unrealistic for this dataset |
| Random Forest (Ensemble) | 1. Highest accuracy (0.995).<br/>2. Perfect precision (1.000).<br/>3. Very high recall (0.986).<br/>4. MCC = 0.989 ==> Very strong balanced performance<br/>5. AUC = 1.000 ==> Perfect class separation. |
| XGBoost (Ensemble) | 1. Identical performance to Random Forest<br/>2. Highest accuracy (0.995).<br/>3. Perfect precision (1.000).<br/>4. Very high recall (0.986).<br/>5. MCC = 0.989 ==> Very strong balanced performance<br/>6. AUC = 1.000 ==> Perfect class separation. 

## Deployment
Deployed on Streamlit Community Cloud.


