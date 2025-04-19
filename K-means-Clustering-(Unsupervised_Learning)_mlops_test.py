import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
import mlflow
import mlflow.sklearn
from fastapi import FastAPI
import uvicorn

# Step 1: Data Ingestion
df = pd.read_csv('myhousepricelist.csv')

# Step 2: Feature Engineering
X = df.drop(columns=['PRICE'])

# Step 3: Train-Test Split
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Step 4: Hyperparameter Tuning
model = KMeans()
param_grid = {
    'n_clusters': [3, 4, 5],
    'init': ['k-means++', 'random']
}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='silhouette_score')
grid_search.fit(X_train)
best_model = grid_search.best_estimator_

# Step 5: Experiment Tracking with MLflow
with mlflow.start_run():
    best_model.fit(X_train)
    y_pred = best_model.predict(X_test)
    silhouette = silhouette_score(X_test, y_pred)
    mlflow.log_metric("silhouette_score", silhouette)
    mlflow.sklearn.log_model(best_model, "model")
    predictions_df = pd.DataFrame({'Predicted Cluster': y_pred})
    predictions_df.to_csv('predictions.csv', index=False)
    mlflow.log_artifact("predictions.csv")
    print("Model and predictions logged successfully.")
    print(f"Silhouette Score: {silhouette}")

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

@app.post
