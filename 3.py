import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

# Step 1: Load the dataset
columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

# Load dataset
data = pd.read_csv('C:/Users/adity/Desktop/pro/project 3/heart.csv', header=None, names=columns)

# Remove invalid rows where the data contains column headers as a row
data = data[data['age'] != 'age']

# Convert columns to numeric
data = data.apply(pd.to_numeric, errors='coerce')

# Drop rows with missing target values or features
data = data.dropna()

# Debugging: Check dataset after cleanup
print("Dataset after cleanup:")
print(data.head())
print("\nClass Distribution:")
print(data['target'].value_counts())

# Step 2: Split the data into features (X) and target (y)
X = data.drop(columns=['target'])
y = data['target']

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train a machine learning model
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train_scaled, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test_scaled)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy on Test Data:", accuracy_score(y_test, y_pred))

# Step 7: Interpret the model using LIME
explainer = LimeTabularExplainer(
    training_data=X_train_scaled,
    feature_names=X.columns,
    class_names=['0', '1'],  # Assuming binary classification
    mode='classification'
)

# Select an instance to explain
instance_idx = 0
instance = X_test_scaled[instance_idx].reshape(1, -1)
exp = explainer.explain_instance(
    data_row=X_test_scaled[instance_idx],
    predict_fn=model.predict_proba
)

# Visualize explanation
exp.show_in_notebook(show_table=True, show_all=False)

# Save explanation as an HTML file (Optional)
exp.save_to_file('lime_explanation.html')

# Global feature importance (based on model's internal feature importances)
importances = model.feature_importances_
sorted_indices = np.argsort(importances)[::-1]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[sorted_indices], align='center')
plt.xticks(range(len(importances)), X.columns[sorted_indices], rotation=90)
plt.title("Feature Importances (Global)")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()
