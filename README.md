# Customer Churn Prediction

---

## Introduction

This project develops and evaluates machine learning models to predict customer churn in a SaaS environment. Churn prediction is formulated as a supervised binary classification problem, where customers are classified as **churned (1)** or **retained (0)** based on behavioral, financial, and support-related features.

The dataset contains 7,000+ customer records with variables such as tenure, monthly logins, average session time, customer satisfaction (CSAT), pricing changes, payment failures, support tickets, contract type, and revenue metrics.

### Project Objectives

1. Compare multiple classification models under class imbalance  
2. Identify the most influential predictors of churn  
3. Translate predictive insights into actionable business implications  

---

## Data Summary

- **Total observations:** 7,000+  
- **Target variable:** Binary churn indicator  
- **Class distribution:** ~10% churn, ~90% retained  

The dataset contains both numerical and categorical variables representing engagement behavior, financial activity, and customer experience metrics.

Due to class imbalance, evaluation prioritizes **ROC-AUC, precision, recall, and confusion matrix analysis**, rather than relying solely on accuracy.

---

## Data Preprocessing

- Missing value handling  
- One-hot encoding for categorical variables  
- Feature scaling (where required)  
- Stratified train/test split to preserve churn distribution  

---

## Model Development

The following models were implemented and compared:

- Logistic Regression  
- Naive Bayes  
- Random Forest  
- Gradient Boosting variants  

Each model was trained using consistent preprocessing pipelines to ensure fair comparison.

---

## Model Comparison

| Model               | Validation Accuracy | ROC-AUC |
|--------------------|--------------------|---------|
| Logistic Regression | ~85%               | ~0.78   |
| Naive Bayes         | ~82%               | ~0.74   |
| Gradient Boosting   | ~87%               | ~0.80   |
| **Random Forest**   | **88%**            | **0.81**|

The **Random Forest classifier** demonstrated the strongest overall performance, particularly in identifying minority churn cases while maintaining balanced precision and recall.

---

## Confusion Matrix Analysis

Confusion matrices were used to evaluate classification performance beyond aggregate metrics.

Key observations:

- High true negative rate for retained customers  
- Reduced false negatives (missed churn cases) in tree-based models  
- Logistic Regression showed stronger sensitivity to pricing variables but weaker capture of non-linear engagement behavior  

This analysis confirmed that Random Forest provided the most balanced classification performance.

---

## Feature Importance & Interpretation

Feature importance analysis from the Random Forest model identified the strongest churn predictors:

- Customer Satisfaction (CSAT)  
- Tenure (Months)  
- Monthly Logins  
- Average Session Time  
- Payment Failures  

Interestingly, total revenue showed relatively low predictive power compared to behavioral engagement signals. This suggests churn is more strongly influenced by user experience and activity patterns than by overall spending levels.

Tree-based modeling provided more reliable interpretability compared to linear coefficients, as it captures complex, non-linear feature interactions.

---

## Business Implications

The findings support a proactive churn management strategy:

- Engagement decline serves as an early warning indicator  
- Low satisfaction scores warrant targeted intervention  
- Newer customers require structured onboarding support  

---

## Repository Structure

- `/data` – dataset (if shareable)  
- `/notebooks` – EDA and model development  
- `/src` – preprocessing and training scripts  
- `/models` – saved model artifacts  
- `/visualizations` – feature importance and confusion matrices  

---

## Future Improvements

- Implement SHAP for advanced interpretability  
- Apply cross-validation for robustness testing  
- Explore resampling techniques (SMOTE) for imbalance handling  
- Deploy model as a lightweight prediction API  

---
