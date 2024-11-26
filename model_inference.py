# C:\Users\kaspa\PycharmProjects\GDP\scripts\model_inference.py

import pandas as pd
import numpy as np
import os
import joblib
import argparse

def load_model(model_path):
    """
    Load the trained model pipeline from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model file at '{model_path}' does not exist.")
    model = joblib.load(model_path)
    print(f"Model loaded from '{model_path}'.")
    return model

def load_new_data(data_path):
    """
    Load new data from a CSV file.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The data file at '{data_path}' does not exist.")
    df = pd.read_csv(data_path)
    print(f"New data loaded from '{data_path}'.")
    return df

def prepare_new_data(df, feature_columns):
    """
    Ensure that the new data contains the required features.
    """
    missing_features = [feature for feature in feature_columns if feature not in df.columns]
    if missing_features:
        raise ValueError(f"The following required features are missing from the input data: {missing_features}")
    X_new = df[feature_columns]
    print("New data successfully prepared for prediction.")
    return X_new

def make_predictions(model, X_new):
    """
    Use the trained model to make predictions on new data.
    """
    predictions = model.predict(X_new)
    print("Predictions successfully made.")
    return predictions

def save_predictions(df, predictions, output_path):
    """
    Save the predictions alongside the input data to a new CSV file.
    """
    df_with_predictions = df.copy()
    df_with_predictions['predicted_gdp_next_year'] = predictions
    df_with_predictions.to_csv(output_path, index=False)
    print(f"Predictions saved to '{output_path}'.")

def main():
    parser = argparse.ArgumentParser(description="GDP Prediction Inference Script")
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model pipeline (.pkl file)')
    parser.add_argument('--data', type=str, required=True, help='Path to the new data CSV file')
    parser.add_argument('--output', type=str, required=False, default='predicted_gdp_next_year.csv', help='Path to save the predictions CSV file')

    args = parser.parse_args()

    # Define the feature columns as used during training
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

    # Load the trained model
    model = load_model(args.model)

    # Load new data
    new_data = load_new_data(args.data)

    # Prepare new data
    X_new = prepare_new_data(new_data, feature_columns)

    # Make predictions
    predictions = make_predictions(model, X_new)

    # Save predictions
    save_predictions(new_data, predictions, args.output)

if __name__ == "__main__":
    main()
