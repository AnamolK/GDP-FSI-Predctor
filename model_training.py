# C:\Users\kaspa\PycharmProjects\GDP\scripts\model_training.py

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure consistent plotting style
sns.set(style="whitegrid")

def load_merged_data(filepath):
    """
    Load merged data from CSV.

    Parameters:
    - filepath: str, path to the merged CSV file.

    Returns:
    - df: DataFrame, merged dataset with features and target.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Merged data loaded from '{filepath}'.")
        return df
    except Exception as e:
        print(f"Error loading merged data: {e}")
        return None

def prepare_features_and_target(df):
    """
    Prepare feature matrix X and target vector y for regression.

    Parameters:
    - df: DataFrame, merged dataset with features and target.

    Returns:
    - X: DataFrame, feature matrix.
    - y: Series, target vector.
    """
    # Define feature columns
    feature_columns = [
        'C1: Security Apparatus',
        'C2: Factionalized Elites',
        'C3: Group Grievance',
        'E1: Economy',
        'E2: Economic Inequality',
        'E3: Human Flight and Brain Drain',
        'P1: State Legitimacy',
        'P2: Public Services',
        'P3: Human Rights',
        'S1: Demographic Pressures',
        'S2: Refugees and IDPs',
        'X1: External Intervention',
        'gdp'  # Current year's GDP as a feature
    ]

    # Check if all feature columns are present
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        print(f"Error: Missing feature columns in merged data: {missing_features}")
        return None, None

    # Define feature matrix X and target vector y
    X = df[feature_columns]
    y = df['gdp_next_year']

    print("\nFeatures and target successfully prepared.")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")

    return X, y

def explore_data(X, y, plots_dir):
    """
    Perform exploratory data analysis for regression.

    Parameters:
    - X: DataFrame, feature matrix.
    - y: Series, target vector.
    - plots_dir: str, directory to save plots.
    """
    print("\n--- Exploratory Data Analysis ---\n")

    print("Target Variable Distribution:")
    print(y.describe())

    plt.figure(figsize=(8, 6))
    sns.histplot(y, kde=True, bins=30, color='skyblue')
    plt.title('Distribution of GDP Next Year')
    plt.xlabel('GDP Next Year')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'gdp_next_year_distribution.png'))
    plt.close()
    print("GDP Next Year distribution plot saved.")

    print("\nFeature Correlation Matrix:")
    corr_matrix = X.corr()
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'feature_correlation_matrix.png'))
    plt.close()
    print("Feature correlation matrix plot saved.")

def build_pipeline(model_type='random_forest'):
    """
    Build a machine learning pipeline with scaling and regression.

    Parameters:
    - model_type: str, type of regression model ('random_forest' or 'gradient_boosting').

    Returns:
    - pipeline: Pipeline object.
    """
    if model_type == 'random_forest':
        regressor = RandomForestRegressor(random_state=42)
    elif model_type == 'gradient_boosting':
        regressor = GradientBoostingRegressor(random_state=42)
    else:
        raise ValueError("Unsupported model type. Choose 'random_forest' or 'gradient_boosting'.")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', regressor)
    ])
    return pipeline

def perform_grid_search(pipeline, X_train, y_train):
    """
    Perform hyperparameter tuning using GridSearchCV for regression.

    Parameters:
    - pipeline: Pipeline object.
    - X_train: DataFrame, training feature matrix.
    - y_train: Series, training target vector.

    Returns:
    - best_estimator: Pipeline object with best parameters.
    """
    if isinstance(pipeline.named_steps['regressor'], RandomForestRegressor):
        param_grid = {
            'regressor__n_estimators': [100, 200, 300],
            'regressor__max_depth': [None, 10, 20, 30],
            'regressor__min_samples_split': [2, 5, 10],
            'regressor__min_samples_leaf': [1, 2, 4],
            'regressor__bootstrap': [True, False]
        }
    elif isinstance(pipeline.named_steps['regressor'], GradientBoostingRegressor):
        param_grid = {
            'regressor__n_estimators': [100, 200, 300],
            'regressor__learning_rate': [0.01, 0.1, 0.2],
            'regressor__max_depth': [3, 5, 7],
            'regressor__min_samples_split': [2, 5, 10],
            'regressor__min_samples_leaf': [1, 2, 4]
        }
    else:
        param_grid = {}

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=2,
        scoring='r2',  # Using R² as the scoring metric
        error_score='raise'  # Raises error if any fit fails
    )

    print("\nStarting Grid Search for Hyperparameter Tuning...")
    grid_search.fit(X_train, y_train)
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best R² Score from Grid Search: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test, plots_dir):
    """
    Evaluate the trained regression model on the test set.

    Parameters:
    - model: Pipeline object, trained regression model.
    - X_test: DataFrame, testing feature matrix.
    - y_test: Series, testing target vector.
    - plots_dir: str, directory to save plots.
    """
    y_pred = model.predict(X_test)

    # Evaluation Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print("\n--- Model Evaluation Metrics ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    # Residual Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residuals vs. Predicted GDP Next Year')
    plt.xlabel('Predicted GDP Next Year')
    plt.ylabel('Residuals')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'residuals_plot.png'))
    plt.close()
    print("Residuals plot saved.")

    # Actual vs Predicted Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title('Actual vs. Predicted GDP Next Year')
    plt.xlabel('Actual GDP Next Year')
    plt.ylabel('Predicted GDP Next Year')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'actual_vs_predicted.png'))
    plt.close()
    print("Actual vs. Predicted plot saved.")

def plot_feature_importance(model, feature_names, plots_dir):
    """
    Plot feature importances from the trained regression model.

    Parameters:
    - model: Pipeline object, trained regression model.
    - feature_names: list, names of the features.
    - plots_dir: str, directory to save plots.
    """
    importances = model.named_steps['regressor'].feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 8))
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], palette='viridis')
    plt.title('Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'feature_importances.png'))
    plt.close()

    print("\n--- Feature Importances ---")
    for i, feature in enumerate(np.array(feature_names)[indices]):
        print(f"{i + 1}. {feature}: {importances[indices][i]:.4f}")

def cross_validate_model(model, X_train, y_train):
    """
    Perform cross-validation and print scores.

    Parameters:
    - model: Pipeline object, trained regression model.
    - X_train: DataFrame, training feature matrix.
    - y_train: Series, training target vector.
    """
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2', n_jobs=-1)
    print(f"\n--- Cross-Validation R² Scores ---")
    print(f"Scores: {cv_scores}")
    print(f"Mean R² Score: {cv_scores.mean():.4f}")
    print(f"Standard Deviation: {cv_scores.std():.4f}")

def save_trained_model(model, models_dir):
    """
    Save the trained model pipeline to disk.

    Parameters:
    - model: Pipeline object, trained regression model.
    - models_dir: str, directory to save the model.
    """
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'trained_model_pipeline.pkl')
    joblib.dump(model, model_path)
    print(f"\nTrained model pipeline saved to '{model_path}'.")

def main():
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(script_dir, '..', 'plots')
    models_dir = os.path.join(script_dir, '..', 'models')
    merged_data_path = r'C:\Users\kaspa\PycharmProjects\GDP\data\merged_data.csv'

    # Create directories if they don't exist
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Step 1: Load merged data
    df = load_merged_data(merged_data_path)
    if df is None:
        print("Failed to load merged data.")
        return

    # Step 2: Prepare features and target
    X, y = prepare_features_and_target(df)
    if X is None or y is None:
        print("Failed to prepare features and target.")
        return

    # Step 3: Exploratory Data Analysis
    explore_data(X, y, plots_dir)

    # Step 4: Build Pipeline
    pipeline = build_pipeline(model_type='random_forest')  # Options: 'random_forest', 'gradient_boosting'

    # Step 5: Hyperparameter Tuning
    best_model = perform_grid_search(pipeline, X, y)

    # Step 6: Split Data for Evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # Step 7: Retrain Best Model on Training Data
    print("\nRetraining the best model on the entire training set...")
    best_model.fit(X_train, y_train)

    # Step 8: Model Evaluation
    evaluate_model(best_model, X_test, y_test, plots_dir)

    # Step 9: Feature Importance
    plot_feature_importance(best_model, X.columns.tolist(), plots_dir)

    # Step 10: Cross-Validation
    cross_validate_model(best_model, X_train, y_train)

    # Step 11: Save Trained Model
    save_trained_model(best_model, models_dir)

if __name__ == "__main__":
    main()
