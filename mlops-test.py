import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import IsolationForest
import mlflow
import mlflow.sklearn
from fastapi import FastAPI
import uvicorn

# Step 1: Data Ingestion
# Load the dataset from CSV file
df = pd.read_csv('myhousepricelist.csv')

# Step 2: Feature Engineering
# Simulate features similar to the Boston housing dataset
# Assuming the CSV has columns like 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'PRICE'
X = df.drop(columns=['PRICE'])
y = df['PRICE']

# Step 3: Train-Test Split
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Hyperparameter Tuning
# Define the model and hyperparameters
model = LinearRegression()
param_grid = {
    'fit_intercept': [True, False],
    'normalize': [True, False]
}

# Perform Grid Search
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Step 5: Experiment Tracking with MLflow
# Start an MLflow run
with mlflow.start_run():
    # Train the best model
    best_model.fit(X_train, y_train)
    
    # Step 6: Prediction
    # Make predictions on the test set
    y_pred = best_model.predict(X_test)
    
    # Step 7: Evaluation
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Log metrics
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    
    # Log model
    mlflow.sklearn.log_model(best_model, "model")
    
    # Save predictions to a CSV file
    predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    predictions_df.to_csv('predictions.csv', index=False)
    
    # Log the predictions file
    mlflow.log_artifact("predictions.csv")
    
    print("Model and predictions logged successfully.")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R^2 Score: {r2}")

# Step 8: Model and Data Drift Detection
# Using Isolation Forest for data drift detection
drift_detector = IsolationForest(contamination=0.1)
drift_detector.fit(X_train)

# Check for data drift in the test set
drift_predictions = drift_detector.predict(X_test)
drifted_data = X_test[drift_predictions == -1]

print(f"Number of drifted data points: {len(drifted_data)}")

# Step 9: Inference
# Load the model from MLflow
model_uri = "runs:/<run_id>/model"  # Replace <run_id> with the actual run ID
loaded_model = mlflow.sklearn.load_model(model_uri)

# Make predictions with the loaded model
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
