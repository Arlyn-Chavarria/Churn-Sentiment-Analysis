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

warnings.filterwarnings("ignore")
#Reading Data
df = pd.read_csv("customer_churn_business_dataset.csv")

#Settng up data before splitting to train vs test

X = df

X = pd.get_dummies(X, columns = ["contract_type"], drop_first= True)

X["price_increase_last_3m"] = X["price_increase_last_3m"].map({
    "No": 0,
    "Yes": 1,
})

X = X[["tenure_months", "contract_type_Quarterly", "contract_type_Yearly", "monthly_logins","weekly_active_days", "avg_session_time", "features_used", "monthly_fee", "total_revenue", "price_increase_last_3m", "support_tickets"]]
Y = df[["churn"]]

# splitting my training data and test data

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    Y, 
    test_size=0.3, 
    random_state=42)

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Fun test not knowing what causes churn
print("test 1: " + str(model.predict([[6, False, False, 0, 0, 0, 0, 35, 0, 0, 5]])))

print("Accuracy:", accuracy_score(y_test, y_pred))
y_proba = model.predict_proba(X_test)[:,1]
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))


w0 = model.intercept_[0]
w = model.coef_[0]

coef_df = pd.DataFrame({
    "feature": X.columns,
    "coefficient": w
})

coef_df["abs_coef"] = np.abs(coef_df["coefficient"])
coef_df = coef_df.sort_values(by="abs_coef", ascending=False)

#print(coef_df)

# | Feature                   | Coefficient | Effect                                                |
# | ------------------------- | ----------- | ----------------------------------------------------- |
# | `tenure_months`           | -0.369      | Longer tenure → less likely to churn                  |
# | `monthly_logins`          | -0.303      | More frequent logins → less likely to churn           |
# | `price_increase_last_3m`  | 0.057       | Recent price increase → slightly more likely to churn |
# | `features_used`           | 0.043       | Using more features → slightly more likely to churn   |
# | `support_tickets`         | -0.041      | More tickets → slightly less likely to churn          |
# | `contract_type_Yearly`    | -0.028      | Yearly contract → less likely to churn                |
# | `contract_type_Quarterly` | -0.028      | Quarterly contract → less likely to churn             |
# | `avg_session_time`        | -0.026      | Longer sessions → less likely to churn                |
# | `total_revenue`           | -0.023      | Higher revenue → slightly less likely to churn        |
# | `weekly_active_days`      | 0.021       | More active days → slightly more likely to churn      |
# | `monthly_fee`             | -0.010      | Higher monthly fee → very small effect                |

#Tenure and engagement matter most

# 1. Longer tenure (tenure_months) and higher logins (monthly_logins) strongly reduce churn probability.
#       These are your primary retention signals.
# 2. Price changes influence churn
#       price_increase_last_3m and Features used has a positive coefficient → price hikes slightly increase churn risk.
# 3. Contract type matters, but weakly
#       Yearly/Quarterly/ contracts slightly reduce churn compared to baseline (baseline - Monthly)
# 4. Minor contributors
#       weekly active days, revenue, monthly fee have very small weights → not very influential individually.

#tests knowing what causes churn
print("test 2: " + str(model.predict([[5, False, False, 0, 100, 0, 15, 50, 0, 1, 0]])))

#Naive Bayes Now

nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

feature_diff = pd.DataFrame({
    "feature": X.columns,
    "mean_diff": nb.theta_[1] - nb.theta_[0],  # class 1 mean - class 0 mean
    "std_class0": nb.var_[0],
    "std_class1": nb.var_[0]
})

feature_diff["effect_size"] = feature_diff["mean_diff"] / np.sqrt(feature_diff["std_class0"] + feature_diff["std_class1"])

feature_diff = feature_diff.reindex(feature_diff["effect_size"].abs().sort_values(ascending=False).index)

y_proba = nb.predict_proba(X_test)[:,1]

print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))
#print(feature_diff)

#same results as logistic regression
# Why GaussianNB accuracy is high
# Class imbalance dominates accuracy
# Predicting “not churn” most of the time will give ~90% raw accuracy
# GaussianNB tends to fit the majority class distribution well
# Even if it completely misses the minority class, raw accuracy will still look very high
# GaussianNB assumption matches your features
# GaussianNB assumes each feature is normally distributed per class
# Many of your numeric features (tenure, monthly logins, avg session time) roughly follow a distribution → NB can model class 0 (not churn) very well
# Naive independence assumption
#NB multiplies probabilities of each feature assuming independence
# This often works surprisingly well for majority class prediction
# Minority class prediction may still be poor, but overall accuracy looks “high”
#before we do Categorical Naive Bayes, we need to transform the data

X2 = df[["tenure_months", "contract_type", "monthly_logins","weekly_active_days", "avg_session_time", "features_used", "monthly_fee", "total_revenue", "price_increase_last_3m", "support_tickets"]]
X2['contract_type'] = LabelEncoder().fit_transform(X2['contract_type'])
X2["price_increase_last_3m"] = X2["price_increase_last_3m"].map({
    "No": 0,
    "Yes": 1,
})

X_train, X_test, y_train, y_test = train_test_split(
    X2, 
    Y, 
    test_size=0.3, 
    random_state=42)

cnb = CategoricalNB()
cnb.fit(X_train, y_train)
y_pred = cnb.predict(X_test)
y_proba = cnb.predict_proba(X_test)[:,1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

n_features, n_categories = 2,60 # classes = churn and not churn, 
results = []
feature_names = X_train.columns.tolist()

for j in range(n_features):
    for k in range(n_categories):
        log_prob_class0 = cnb.feature_log_prob_[0][j][k]
        log_prob_class1 = cnb.feature_log_prob_[0][j][k]

        log_odds_diff = log_prob_class1 - log_prob_class0  # Positive → favors churn

        results.append({
            "feature": feature_names[j],
            "category_index": k,
            "log_prob_class0": log_prob_class0,
            "log_prob_class1": log_prob_class1,
            "log_odds_diff": log_odds_diff
        })

weights_df = pd.DataFrame(results)
weights_df = weights_df.reindex(weights_df["log_odds_diff"].abs().sort_values(ascending=False).index)
pd.set_option('display.max_rows', None)
#print(weights_df)

#categorical is not good for this test. because the data is 90% no churn, it can easily predict which one. that is why accuracy is so high
#High accuracy for your CategoricalNB is mostly because of class imbalance.
#It doesn’t mean it’s a better model — Logistic Regression might actually capture minority churn better, even if raw accuracy is lower.

categorical_features = ["contract_type", "price_increase_last_3m"]
numeric_features = [
    "tenure_months", "monthly_logins","weekly_active_days",
    "avg_session_time", "features_used", "monthly_fee", "total_revenue", "support_tickets"
]
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight='balanced',  # handle class imbalance
        random_state=42
    ))
])

rf_pipeline.fit(X_train, y_train)

# --- 6. Evaluate ---
y_pred = rf_pipeline.predict(X_test)
y_proba = rf_pipeline.predict_proba(X_test)[:,1]

print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))
      

rf_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight='balanced',  # handle class imbalance
        random_state=42
    ))
])

rf_pipeline.fit(X_train, y_train)

# --- 6. Evaluate ---
y_pred = rf_pipeline.predict(X_test)
y_proba = rf_pipeline.predict_proba(X_test)[:,1]

print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))