# diabetes_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------
# 1. Load dataset
# -------------------------
data = pd.read_csv("diabetes_risk_dataset.csv")  # Make sure the CSV is in the same folder

print("First 5 rows of dataset:")
print(data.head())
print("\nDataset info:")
print(data.info())

# Check for missing values
print("\nMissing values per column:")
print(data.isnull().sum())

# -------------------------
# 2. Basic preprocessing
# -------------------------
# Fill missing values with median (if any)
data = data.fillna(data.median())

# Separate features and target
X = data.drop("Outcome", axis=1)  # Adjust "Outcome" if your target column is different
y = data["Outcome"]

# -------------------------
# 3. Train/Test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# 4. Train Logistic Regression
# -------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------------
# 5. Evaluate Model
# -------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)

# -------------------------
# 6. Short summary
# -------------------------
print("\nSummary:")
print(f"The logistic regression model achieved an accuracy of {accuracy:.2f} on the test set.")
print("The confusion matrix and classification report show prediction performance for positive and negative cases.")