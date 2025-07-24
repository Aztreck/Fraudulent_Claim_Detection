# 🚩 Fraud Detection Using Logistic Regression

This project focuses on detecting insurance fraud using a logistic regression model. It explores categorical and numerical features to predict the likelihood of a fraud report (`fraud_reported`) based on claim and customer-related attributes.

---

## 📊 Dataset Overview

The dataset includes 999 insurance claims with a binary target variable `fraud_reported`. The features span across demographics, policy details, claim statistics, and incident attributes.

### 🔢 Key Numerical Features:
- `months_as_customer`
- `policy_deductable`
- `capital-gains`
- `capital-loss`
- `incident_hour_of_the_day`
- `number_of_vehicles_involved`
- `bodily_injuries`
- `witnesses`
- `total_claim_amount`
- `injury_claim`
- `property_claim`
- `vehicle_claim`
- `auto_year`
- `customer_risk_score` *(engineered)*
- `sum_sub_claims` *(engineered from injury + property + vehicle claim)*

### 🏷️ Key Categorical Features (One-Hot Encoded):
- `policy_state`
- `insured_education_level`
- `insured_occupation`
- `insured_relationship`
- `incident_type`
- `incident_severity`
- `authorities_contacted`
- `incident_state`
- `incident_city`
- `police_report_available`
- `hobby_bucket`
- `auto_make_bucket`

---

## 🔍 Exploratory Data Analysis

### Categorical Target Event Rates:
- High fraud rates were observed for:
  - `incident_type` = "Single Vehicle Collision"
  - `insured_occupation` = "exec-managerial"
  - `insured_relationship` = "other-relative"
  - `hobby_bucket` = "Mental/Indoor"

These categorical variables demonstrated meaningful variation in fraud probability.

---

## 🧠 Feature Engineering

- `sum_sub_claims` = `injury_claim + property_claim + vehicle_claim`
- `customer_risk_score`: A derived feature using claim behavior
- `incident_date`: Extracted components like month or day if needed
- Applied extensive **one-hot encoding** on categorical variables

---

## 📌 Multicollinearity Check

**VIF (Variance Inflation Factor)** was calculated to identify highly collinear features.

- Removed high-VIF features such as:
  - `sum_sub_claims`
  - `vehicle_claim` (if correlated heavily with others)
- Final model used features with **VIF < 5** to ensure stability

---

## 🧪 Model Building: Logistic Regression

- Model: `LogisticRegression(solver='liblinear')`
- Selected for binary classification with small dataset
- Feature selection: **RFECV (Recursive Feature Elimination with Cross-Validation)**

### RFECV Insight:
Initially, RFECV chose only **1 feature** due to imbalance and high correlation. Post-balancing and feature engineering, ~30 features were retained.

---

## 📈 Model Evaluation

### 🔹 ROC Curve
- **AUC = 0.87**
- Strong discriminative power

### 🔹 Accuracy / Sensitivity / Specificity vs. Threshold
- Optimal trade-off around **threshold = 0.5**
- Lower thresholds: Higher sensitivity (catch more fraud)
- Higher thresholds: Higher specificity (fewer false alarms)

### 🔹 Precision vs. Recall
- Typical inverse trade-off
- For fraud detection, higher **recall** is preferred to avoid missing fraudulent claims

---

## 📊 GLM Model Summary (Top Insights)

| Feature                              | Coef  | Insight                                      |
|--------------------------------------|--------|-----------------------------------------------|
| `witnesses`                          | +0.268 | More witnesses → Higher fraud chance          |
| `vehicle_claim`                      | +0.499 | Higher vehicle claim → Higher fraud risk      |
| `injury_claim`                       | -0.223 | Lower injury claims may be more suspicious    |
| `customer_risk_score`               | +0.298 | Strong predictor of fraud                     |
| `insured_education_level_Masters`   | -0.244 | Higher education correlates with lower fraud  |
| `incident_city_Northbrook`          | -0.309 | Indicates safer region                        |
| `hobby_bucket_Mental/Indoor`        | +0.294 | Indoor hobbies → Higher likelihood of fraud   |

_(P-values < 0.05 considered significant)_

---

## ✅ Summary

- Logistic Regression achieved **AUC = 0.87**
- Key predictors: claims, witnesses, customer risk score, and categorical demographics
- Feature selection, VIF filtering, and one-hot encoding improved model quality
- Threshold tuning is essential to balance fraud detection with false positive control

---

## 📁 Project Structure

- zip file containing the solution