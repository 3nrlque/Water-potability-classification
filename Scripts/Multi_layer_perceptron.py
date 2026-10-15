import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import collections
import seaborn as sns 
import matplotlib.pyplot as plt
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15,random_state=SEED)

# Standardize the input features using statistics calculated from the training set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Display class distribution before SMOTE
print(f"Class distribution in original test set: {collections.Counter(y_train)}")
plt.figure(figsize=(7, 5))
sns.countplot(x=y_train, palette='viridis')
plt.title('Class distribution before SMOTE')
plt.xlabel('Potability')
plt.ylabel('Number of samples')
plt.xticks([0, 1], ['non-potable', 'potable'])
plt.show()

# Fix class imbalance using SMOTE
sm = SMOTE(sampling_strategy='minority',random_state=SEED)
X_train_balanced, y_train_balanced = sm.fit_resample(X_train_scaled, y_train)

# Display class distribution after SMOTE
print(f"Class distribution before SMOTE: {collections.Counter(y_train_balanced)}")
plt.figure(figsize=(7, 5))
sns.countplot(x=y_train_balanced, palette='viridis')
plt.title('Class distribution after SMOTE')
plt.xlabel('Potability')
plt.ylabel('Number of samples')
plt.xticks([0, 1], ['non-potable', 'potable'])
plt.show()

# Multi-Layer Perceptron model
mlp = MLPClassifier(
    hidden_layer_sizes=(64,32,16),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=SEED,
    early_stopping=True, 
    validation_fraction=0.1765,
    n_iter_no_change=10
)
mlp.fit(X_train_balanced, y_train_balanced)

# Save the trained model
#dump(mlp, "MLP_64-32-16_model.pkl")
#dump(scaler, "Standard_scaler.pkl")

# Generate predictions for the test set
y_pred = mlp.predict(X_test_scaled)

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
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Real')
plt.title('Confusion Matrix – MLP 64-32-16')
plt.show()