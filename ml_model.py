# ml_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load and clean
df = pd.read_csv("pricerunner_aggregate.csv")
df.columns = df.columns.str.strip()
df = df.drop_duplicates().dropna()

# Sample to reduce memory usage
df = df.sample(n=10000, random_state=42)

# Features and target
X = df["Product Title"]
y = df["Cluster Label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline with Logistic Regression
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=3000, stop_words='english')),
    ('clf', LogisticRegression(max_iter=1000))
])

# Train model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "model.pkl")
