# Credit Card Fraud Detection

This project uses machine learning to detect and analyze fraudulent credit card transactions in a highly imbalanced financial dataset. The project focuses on fraud classification, class imbalance handling, clustering analysis, and model explainability to better understand transaction patterns associated with fraudulent activity.

## Project Files

- [Final Project Report](./Credit_Card_Fraud_Report.pdf)
- [Jupyter Notebook](./Credit_Card_Fraud_Detection.ipynb)

## Project Source

Intro to Data Science Course Project

## Dataset

The project uses the Kaggle Credit Card Fraud Detection dataset, which contains anonymized real-world credit card transaction data. Most features were transformed using PCA to protect customer privacy, while `Time` and `Amount` were scaled during preprocessing.

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
2. Performed exploratory data analysis to evaluate class imbalance and transaction behavior.
3. Scaled the `Time` and `Amount` variables during preprocessing.
4. Split the dataset into training and testing sets using stratified sampling.
5. Applied SMOTE to address severe class imbalance within the training data.
6. Used PCA and K-Means clustering to group transaction behavior patterns and identify higher-risk clusters.
7. Trained Logistic Regression and XGBoost classification models.
8. Evaluated model performance using accuracy, precision, recall, F1-score, ROC curves, and precision-recall curves.
9. Applied SHAP analysis to identify the most influential variables contributing to fraud predictions.

## Key Findings

- Fraud represented only about 0.173% of all transactions, creating a severe class imbalance problem.
- SMOTE helped balance the training data by generating synthetic fraud observations.
- Logistic Regression achieved stronger fraud recall but generated more false positives.
- XGBoost achieved higher overall accuracy and stronger fraud precision while still maintaining strong fraud detection performance.
- SHAP analysis identified anonymized variables such as `V4` and `V14` as highly influential predictors of fraud.
- In fraud detection systems, recall is especially important because failing to identify fraudulent transactions is typically more costly than generating false alerts.

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

Credit card fraud detection is a high-impact problem for banks, payment processors, fintech companies, and risk management teams. This project demonstrates how machine learning can be used to flag suspicious transactions, reduce false negatives, and support fraud prevention and transaction monitoring workflows.

## Limitations

- The dataset uses anonymized features, limiting business interpretation of individual transaction variables.
- Fraud observations are extremely rare, requiring synthetic oversampling with SMOTE.
- The clustering analysis is influenced by SMOTE-generated observations and should not be interpreted as real-world fraud distributions.
- Random Forest was considered but not included because of runtime limitations on the full dataset.
- The project focuses on model analysis rather than real-time production deployment.

## Conclusion

This project demonstrates how machine learning techniques can be applied to detect rare fraudulent transactions within financial data. By combining SMOTE oversampling, clustering analysis, supervised classification models, and SHAP explainability, the project provides both predictive performance and interpretable insight into fraud-related transaction behavior.
