import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# STEP 1 — Load Dataset
# ===============================
print("Loading dataset...")
df = pd.read_csv("train.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# Drop Loan_ID (not useful for prediction)
df = df.drop("Loan_ID", axis=1)

# ===============================
# STEP 2 — EDA Analysis
# ===============================
print("\nPerforming EDA...")

print("\nMissing Values:")
print(df.isnull().sum())

# Loan approval distribution
plt.figure()
sns.countplot(x="Loan_Status", data=df)
plt.title("Loan Approval Distribution")
plt.show()

# Credit history impact
plt.figure()
sns.countplot(x="Credit_History", hue="Loan_Status", data=df)
plt.title("Credit History vs Loan Approval")
plt.show()

# ===============================
# STEP 3 — Handle Missing Values
# (KNN Imputer)
# ===============================
print("\nHandling missing values using KNN...")

# Convert categorical columns temporarily
temp_df = df.copy()

categorical_cols = temp_df.select_dtypes(include='object').columns

for col in categorical_cols:
    temp_df[col] = LabelEncoder().fit_transform(temp_df[col].astype(str))

imputer = KNNImputer(n_neighbors=5)
temp_df[:] = imputer.fit_transform(temp_df)

df = temp_df

print("Missing values after imputation:")
print(df.isnull().sum())

# ===============================
# STEP 4 — Feature Engineering
# ===============================
print("\nCreating new features...")

df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['EMI'] = df['LoanAmount'] / df['Loan_Amount_Term']

# ===============================
# STEP 5 — Encoding
# ===============================
print("\nEncoding categorical data...")

encoder = LabelEncoder()

for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# ===============================
# STEP 6 — Train/Test Split
# ===============================
print("\nSplitting dataset...")

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# STEP 7 — Train Random Forest
# ===============================
print("\nTraining Random Forest model...")

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Initial Accuracy:", accuracy)

# ===============================
# STEP 8 — Hyperparameter Tuning
# ===============================
print("\nRunning Hyperparameter Tuning...")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("Best Parameters:", grid_search.best_params_)

# ===============================
# STEP 9 — Cross Validation
# ===============================
print("\nRunning Cross Validation...")

scores = cross_val_score(best_model, X, y, cv=5)

print("Cross Validation Scores:", scores)
print("Average Accuracy:", scores.mean())

# ===============================
# STEP 10 — Save Model
# ===============================
print("\nSaving model...")

os.makedirs("model", exist_ok=True)

joblib.dump(best_model, "C:/Users/Admin/Documents/mec aids project 2/final/model/loan_model.pkl")

print("Model saved successfully!")

# ===============================
# Feature Importance
# ===============================
importance = best_model.feature_importances_

plt.figure(figsize=(10,6))
sns.barplot(x=importance, y=X.columns)
plt.title("Feature Importance")
plt.show()