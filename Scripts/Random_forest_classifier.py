import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
import random

# Set random seeds for reproducibility
SEED = 56
random.seed(SEED)
np.random.seed(SEED)

# Load data from a CSV file
df = pd.read_csv("////")


# Replace missing values with the mean of each column
df.fillna(df.mean(numeric_only=True), inplace=True)

# Separate the input features from the target variable
X = df.drop('Potability', axis=1)
y = df['Potability']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=SEED)

# Standardize the input features using statistics calculated from the training set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fix class imbalance using SMOTE
sm = SMOTE(sampling_strategy='minority',random_state=SEED)
X_train_balanced, y_train_balanced = sm.fit_resample(X_train_scaled, y_train)

# Update the training variables to use the balanced dataset.
X_train_scaled, y_train = X_train_balanced, y_train_balanced

# Initialize and train the Random Forest classifier
# 300 trees
model = RandomForestClassifier(n_estimators=300,random_state=SEED)
model.fit(X_train_balanced, y_train_balanced)

# Save the trained model
#dump(model, "RF_model.pkl")
#dump(scaler, "Standard_scaler.pkl")

# Generate predictions for the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\nTraining finished")
print(f"Test set accuracy: {accuracy:.4f}")
print("\nClassification report:")
print(report)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Real')
plt.title('Confusion Matrix - Random Forest')
plt.show()
