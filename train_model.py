# train_model.py - Binary classifier (Safe vs Pathogen) with evaluation + confusion matrix plot

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("pathogen_dataset.csv")

# Convert labels to binary
# 0 = Safe, 1/2/3 = Pathogen Present
df['binary_label'] = df['label'].apply(lambda x: 0 if x == 0 else 1)

# Features and target
X = df[['ph', 'turbidity_v', 'temperature_c', 'particles']]
y = df['binary_label']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.joblib")

# Evaluate model
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("✅ Binary classification model trained and saved as model.joblib")
print(f"Accuracy: {acc:.2f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Safe", "Pathogen"]))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plot confusion matrix
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Safe","Pathogen"], yticklabels=["Safe","Pathogen"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Safe vs Pathogen")
plt.tight_layout()
plt.show()
