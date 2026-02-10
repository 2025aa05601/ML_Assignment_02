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

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|-----|---------|-----|----------|-------|----|----|
| Logistic Regression | 0.982 | 0.997 | 0.995 | 0.958 | 0.976 | 0.963 |
| Decision Tree | 0.988 | 0.985 | 0.990 | 0.976 | 0.983 | 0.974 |
| KNN | 0.974 | 0.995 | 0.990 | 0.939 | 0.964 | 0.944 |
| Naive Bayes | 0.626 | 0.885 | 0.455 | 0.024 | 0.045 | 0.024 |
| Random Forest | 0.995 | 1.000 | 1.000 | 0.986 | 0.993 | 0.989 |
| XGBoost | 0.995 | 0.999 | 1.000 | 0.986 | 0.993 | 0.989 |

## Observations

| Model | Observation |
|-----|-------------|
| Logistic Regression | Performs well on linear boundaries |
| Decision Tree | Overfits without pruning |
| KNN | Sensitive to scaling |
| Naive Bayes | Fast but assumes independence |
| Random Forest | Stable & robust |
| XGBoost | Best overall performance |

## Deployment
Deployed on Streamlit Community Cloud.


