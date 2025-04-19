import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import IsolationForest
import mlflow
import mlflow.sklearn
from fastapi import FastAPI
import uvicorn

# Step 1: Data Ingestion
df = pd.read_csv('myhousepricelist.csv')

# Step 2: Feature Engineering
X = df.drop(columns=['PRICE'])
y = (df['PRICE'] > df['PRICE'].median()).astype(int)  # Binary classification based on median price

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Hyperparameter Tuning
model = DecisionTreeClassifier()
param_grid = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Step 5: Experiment Tracking with MLflow
with mlflow.start_run():
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(best_model, "model")
    predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    predictions_df.to_csv('predictions.csv', index=False)
    mlflow.log_artifact("predictions.csv")
    print("Model and predictions logged successfully.")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

# Step 8: Model and Data Drift Detection
drift_detector = IsolationForest(contamination=0.1)
drift_detector.fit(X_train)
drift_predictions = drift_detector.predict(X_test)
drifted_data = X_test[drift_predictions == -1]
print(f"Number of drifted data points: {len(drifted_data)}")

# Step 9: Inference
model_uri = "runs:/<run_id>/model"  # Replace <run_id> with the actual run ID
loaded_model = mlflow.sklearn.load_model(model_uri)
new_predictions = loaded_model.predict(X_test)
print("New predictions:", new_predictions)

# Step 10: Serving the Model using FastAPI
app = FastAPI()

@app.post("/predict")
def predict(features: dict):
    features_df = pd.DataFrame([features])
    prediction = loaded_model.predict(features_df)
    return {"prediction": prediction[0]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
