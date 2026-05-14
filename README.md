# Credit Card Fraud Detection

This project uses machine learning to detect and analyze fraudulent credit card transactions in a highly imbalanced dataset. The goal was to compare fraud detection models, handle rare fraud cases, and identify transaction patterns that may indicate higher fraud risk.

## Project Source

Intro to Data Science Course Project

## Dataset

The project uses the Kaggle Credit Card Fraud Detection dataset, which contains anonymized transaction data from real credit card transactions. Most features were transformed using PCA to protect customer privacy, while `Time` and `Amount` were scaled during preprocessing.

Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Key Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SMOTE
- PCA
- K-Means Clustering
- SHAP
- Matplotlib

## Project Workflow

1. Loaded and reviewed credit card transaction data.
2. Performed exploratory data analysis to understand class imbalance and transaction patterns.
3. Scaled the `Time` and `Amount` variables.
4. Split the dataset into training and testing sets using stratified sampling.
5. Applied SMOTE to address severe class imbalance in the training data.
6. Used PCA and K-Means clustering to group transaction patterns and identify higher-risk clusters.
7. Trained Logistic Regression and XGBoost classification models.
8. Evaluated model performance using accuracy, precision, recall, F1-score, ROC curves, and precision-recall curves.
9. Used SHAP analysis to identify the most important variables contributing to fraud predictions.

## Key Findings

- Fraud represented only about 0.173% of all transactions, creating a severe class imbalance problem.
- SMOTE helped balance the training data by generating synthetic fraud examples.
- Logistic Regression achieved strong fraud recall but produced more false positives.
- XGBoost achieved higher overall accuracy and better precision, while still maintaining strong fraud detection performance.
- SHAP analysis showed that certain anonymized variables, especially `V4` and `V14`, were among the most important predictors of fraud.
- In fraud detection, recall is especially important because missing a fraudulent transaction is usually more costly than flagging a legitimate transaction for review.

## Model Results

### Logistic Regression

- Accuracy: 97.42%
- Fraud Precision: 0.06
- Fraud Recall: 0.92
- Fraud F1-Score: 0.11

### XGBoost

- Accuracy: 99.51%
- Fraud Precision: 0.24
- Fraud Recall: 0.87
- Fraud F1-Score: 0.38

## Skills Demonstrated

- Fraud detection analytics
- Machine learning classification
- Imbalanced data handling
- SMOTE oversampling
- Predictive modeling
- Risk analytics
- Exploratory data analysis
- PCA dimensionality reduction
- K-Means clustering
- Model evaluation
- SHAP explainability
- ETL-style preprocessing
- Financial data analysis

## Business Relevance

Credit card fraud detection is a high-impact problem for banks, payment processors, fintech companies, and risk teams. This project demonstrates how machine learning can be used to flag suspicious transactions, reduce false negatives, and support fraud prevention workflows.

## Limitations

- The dataset uses anonymized features, so individual transaction variables cannot be fully interpreted.
- The fraud class is extremely small, requiring synthetic oversampling with SMOTE.
- The project focuses on model development and analysis, not real-time deployment.
- Random Forest was considered but not used due to runtime limitations on the full dataset.

## Conclusion

This project shows how machine learning can be applied to detect rare fraudulent transactions in financial data. By combining SMOTE, classification models, clustering, and SHAP explainability, the analysis demonstrates both predictive performance and interpretability in a realistic fraud detection setting.
