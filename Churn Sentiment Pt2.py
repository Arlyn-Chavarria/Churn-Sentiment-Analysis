import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import sys

warnings.filterwarnings("ignore")
#Reading Data
df = pd.read_csv("customer_churn_business_dataset.csv")

#separating x and y data

y = df['churn']
x = df[['gender',
'age',
'customer_segment',
'tenure_months',
'signup_channel',
'contract_type',
'monthly_logins',
'weekly_active_days',
'avg_session_time',
'features_used',
'usage_growth_rate',
'last_login_days_ago',
'monthly_fee',
'total_revenue',
'payment_method',
'payment_failures',
'discount_applied',
'price_increase_last_3m',
'support_tickets',
'avg_resolution_time',
'complaint_type',
'csat_score',
'escalations',
'email_open_rate',
'marketing_click_rate',
'nps_score',
'survey_response',
'referral_count']]

#formatting my columns using one-hot encoding or label encoding
#one-hot encoders: Gender,Customer_segment, payment_method, complaint_type
#label encoders: contract_type, survey_response, signup_channel, discount_applied, price_increase_last_3m

#label encoders

x["contract_type"] = x["contract_type"].map({
    "Monthly": 0,
    "Quarterly": 1,
    "Yearly": 2,
})

x["survey_response"] = x["survey_response"].map({
    "Unsatisfied": 0,
    "Neutral": 1,
    "Satisfied": 2,
})

x["signup_channel"] = x["signup_channel"].map({
    "Mobile": 0,
    "Web": 1,
    "Referral": 2,
})

x["discount_applied"] = x["discount_applied"].map({
    'No' : 0,
    'Yes' : 1,
})

x["price_increase_last_3m"] = x["price_increase_last_3m"].map({
    'No' : 0,
    'Yes' : 1,
})

#one hot encoding

x = pd.get_dummies(x, columns=['gender', 'customer_segment', 'payment_method', 'complaint_type'], drop_first=True)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.3, random_state=42)

# for i in [100, 500, 1000, 2000, 2500, 5000, 10000]:
#     lreg = LogisticRegression(class_weight='balanced', max_iter=i).fit(x_train, y_train)

#     y_pred = lreg.predict(x_test)
#     y_proba = lreg.predict_proba(x_test)[:,1]
    # print("\n"+ "iterations: i: " + str(i) + '\n')
    # print("Accuracy: " + str(accuracy_score(y_pred, y_test)))
    # #print("Classification Report:\n", classification_report(y_test, y_pred))
    # print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

lreg = LogisticRegression(class_weight='balanced', max_iter=2500).fit(x_train, y_train)
y_pred = lreg.predict(x_test)
y_proba = lreg.predict_proba(x_test)[:,1]

print("\n"+ "iterations: i: 2500")
print("Accuracy: " + str(accuracy_score(y_pred, y_test)))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

coef_df = pd.DataFrame({
    "feature": x_train.columns,
    "coefficient": lreg.coef_[0]
})

print(coef_df.sort_values(by="coefficient", ascending=False))

#2500 iterations is the sweet spot

# iterations: i: 2500
# Accuracy: 0.6773333333333333
# Classification Report:
#                precision    recall  f1-score   support
#            0       0.93      0.69      0.79      2662
#            1       0.20      0.61      0.30       338
#     accuracy                           0.68      3000
#    macro avg       0.57      0.65      0.54      3000
# weighted avg       0.85      0.68      0.74      3000

# -------------------------
# Logistic Regression Feature Insights For 2500 Iterations
# -------------------------
#
# Positive coefficients → features that INCREASE churn risk:
#   price_increase_last_3m     0.312  → Price increases last 3 months strongly increase churn
#   payment_method_Card        0.276  → Paying by card slightly increases churn
#   customer_segment_Individual 0.253 → Individual customers more likely to churn
#   complaint_type_Technical   0.246  → Technical complaints increase churn
#   marketing_click_rate       0.182  → More marketing clicks slightly increase churn
#   escalations                0.138  → Customers with escalations more likely to churn
#   signup_channel             0.077  → Certain signup channels slightly increase churn
#   payment_method_PayPal      0.075  → Paying via PayPal slightly increases churn
#   complaint_type_Service     0.074  → Service complaints slightly increase churn
#   gender_Male                0.071  → Male customers slightly higher churn
#   usage_growth_rate          0.062  → Rapid usage growth slightly increases churn
#   features_used              0.030  → Using more features slightly increases churn
#   weekly_active_days         0.026  → More active days slightly increase churn
#   referral_count             0.024  → More referrals slightly increase churn
#   last_login_days_ago        0.019  → Less recent login slightly increases churn
#   support_tickets            0.017  → More support tickets slightly increase churn
#   avg_session_time           0.005  → Very minor positive effect
#   avg_resolution_time        0.005  → Very minor positive effect
#   monthly_fee                0.004  → Barely increases churn
#   age                        0.002  → Barely increases churn
#   nps_score                  0.001  → Minimal effect
#
# Negative coefficients → features that DECREASE churn risk:
#   csat_score                 -0.515 → High customer satisfaction greatly reduces churn
#   discount_applied           -0.040 → Customers with discounts slightly less likely to churn
#   survey_response            -0.043 → Responding to surveys slightly reduces churn
#   contract_type              -0.032 → Certain contract types reduce churn (e.g., yearly)
#   monthly_logins             -0.032 → More logins slightly reduce churn
#   tenure_months              -0.022 → Longer tenure reduces churn
#   total_revenue              -0.000 → Barely reduces churn
#
# Notes:
# - Biggest risk factor: price_increase_last_3m
# - Biggest protective factor: csat_score
# - Most other features have minor effects
# - Positive coefficient = increases churn probability
# - Negative coefficient = decreases churn probability

# for i in [100, 500, 1000, 2000, 2500, 5000, 10000]: 
#     randomforest = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators = i).fit(x_train, y_train)
#     y_pred = randomforest.predict(x_test)
#     y_proba = randomforest.predict_proba(x_test)[:,1]

#     print("\n"+ "iterations: i: " + str(i) + '\n')
#     print("Accuracy: " + str(accuracy_score(y_pred, y_test)))
#     #print("Classification Report:\n", classification_report(y_test, y_pred))
#     print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

randomforest = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators = 1000).fit(x_train, y_train)
y_pred = randomforest.predict(x_test)
y_proba = randomforest.predict_proba(x_test)[:,1]

print("\n"+ "iterations: i: 1000")
print("Accuracy: " + str(accuracy_score(y_pred, y_test)))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

# iterations: i: 1000
# Accuracy: 0.887
# Classification Report:
#                precision    recall  f1-score   support
#            0       0.89      1.00      0.94      2662
#            1       0.00      0.00      0.00       338
#     accuracy                           0.89      3000
#    macro avg       0.44      0.50      0.47      3000
# weighted avg       0.79      0.89      0.83      3000

# ROC-AUC Score: 0.8104008197778065

coef_df = pd.DataFrame({
    "feature": x_train.columns,
    "coefficient": randomforest.feature_importances_
})

# Random Forest Feature Importances for Churn Prediction
# ------------------------------------------------------
# Feature                   Importance
# csat_score                 0.111
# tenure_months              0.097
# monthly_logins             0.079
# total_revenue              0.076
# avg_session_time           0.053
# avg_resolution_time        0.052
# payment_failures           0.050
# nps_score                  0.047
# usage_growth_rate          0.047
# email_open_rate            0.047
# last_login_days_ago        0.046
# age                        0.044
# marketing_click_rate       0.043
# features_used              0.028
# weekly_active_days         0.026
# monthly_fee                0.021
# support_tickets            0.019
# referral_count             0.015
# signup_channel             0.013
# survey_response            0.012
# contract_type              0.011
# escalations                0.008
# gender_Male                0.007
# complaint_type_Technical   0.007
# customer_segment_Individual 0.006
# discount_applied           0.006
# price_increase_last_3m     0.006
# payment_method_Card        0.006
# customer_segment_SME       0.006
# payment_method_PayPal      0.006
# complaint_type_Service     0.006
#
# Interpretation:
# - Top predictors: csat_score, tenure_months, monthly_logins, total_revenue
# - Least important: complaint_type, payment method, gender, customer_segment
# - These importance scores show how much the Random Forest model relies on each feature.

#print(coef_df.sort_values(by="coefficient", ascending=False))