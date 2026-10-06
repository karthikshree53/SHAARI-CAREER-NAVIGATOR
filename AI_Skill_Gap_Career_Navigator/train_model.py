import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

# 1. Generate realistic dataset automatically (1000 records)
np.random.seed(42)
n_samples = 1000

python = np.random.randint(30, 100, n_samples)
java = np.random.randint(30, 100, n_samples)
sql = np.random.randint(30, 100, n_samples)
ml = np.random.randint(30, 100, n_samples)
comm = np.random.randint(30, 100, n_samples)

careers = []
for i in range(n_samples):
    if ml[i] > 70 and python[i] > 70:
        careers.append("AI Engineer")
    elif sql[i] > 75 and python[i] > 65:
        careers.append("Data Scientist")
    elif java[i] > 75 and sql[i] > 60:
        careers.append("Software Engineer")
    else:
        careers.append("Data Analyst")

df = pd.DataFrame({
    'python': python, 'java': java, 'sql': sql,
    'ml': ml, 'comm': comm, 'career': careers
})

# Save the dataset so you have a copy in your workspace
df.to_csv('dataset.csv', index=False)
print("[SUCCESS] New dataset.csv generated!")

# 2. Separate features and target
X = df[['python', 'java', 'sql', 'ml', 'comm']]
y = df['career']

# 3. Train-Test Split & Scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 5. Evaluate Model
y_pred = model.predict(X_test_scaled)
print("\n=== MODEL EVALUATION METRICS ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred, average='weighted'):.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 6. Save Model Artifacts
joblib.dump(model, "models/career_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("\n[SUCCESS] Model and Scaler saved into models/ folder!")