# Customer Churn Prediction

---

## Introduction

This project develops and evaluates machine learning models to predict customer churn. I treated churn prediction as a binary classification problem, where customers are classified as **churned (1)** or **retained (0)** based on behavioral, financial, and support-related features.

The dataset contains 7,000+ customer records with variables such as tenure, monthly logins, average session time, customer satisfaction (CSAT), pricing changes, payment failures, support tickets, contract type, and revenue metrics.

### Project Objectives

The objective of this project is to run multiple different data models and determine which model provides the highest accuracy and can detect when a customer is most likely to churn.
Logistic Regression
Gaussian Naive Bayes
Categorical Naive Bayes
Random Forest

---

## Data Source

I gathered our data from kaggle, which is an online community platform, that contains free-to-use data. Here is the link to the data used to train the model: https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset

---

## Data Summary

There are 7000+ total customers with their history on either they have still churned, or did not churn. based on this dataset, ~10% are customers that churned and ~90 are customers that have not churned.

There is huge discrepancy of churn vs not churn customers, however because of this, we will rely on ROC-AUC and not soleley on accuracy.

I acknowledge that the dataset contains both numerical and categorical variables 

We also acknowledge that some metrics defined in this dataset may not bear any influence if a customer is to churn or not.

---

## Data Preprocessing

Before training the models, the dataset was carefully prepared to ensure reliable results. I began by identifying and handling missing values to maintain data consistency. Categorical variables were then converted into numerical form using one-hot encoding so the models could properly interpret them. Where appropriate, feature scaling was applied to keep variables on comparable ranges. Finally, the data was randomly split into training and testing sets using a stratified approach to preserve the original churn distribution, ensuring the models were evaluated on a representative sample of customers. this meant that our training data was 70% of the data and our test was 30%. This also meant that the churned customres we equally split into both training data and the test data.

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
