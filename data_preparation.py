# C:\Users\kaspa\PycharmProjects\GDP\scripts\data_preparation.py

import re
import pandas as pd
import os
from glob import glob
import pycountry

# Define paths to data folders
GDP_FILE_PATH = r'C:\Users\kaspa\PycharmProjects\GDP\data\gdp\WITS-Country-Timeseries.xlsx'
FSI_DATA_FOLDER = r'C:\Users\kaspa\PycharmProjects\GDP\data\fsi'
OUTPUT_MERGED_CSV_PATH = r'C:\Users\kaspa\PycharmProjects\GDP\data\merged_data.csv'


def standardize_country_name(name):
    """
    Standardize country names using pycountry.
    """
    try:
        return pycountry.countries.lookup(name).name
    except LookupError:
        # Handle special cases or return the original name
        special_cases = {
            'Congo Democratic Republic': 'Congo, The Democratic Republic of the',
            'Czech Republic': 'Czechia',
            'United States': 'United States of America',
            'South Korea': 'Korea, Republic of',
            'North Korea': "Korea, Democratic People's Republic of",
            'Syria': 'Syrian Arab Republic',
            'Burma (Myanmar)': 'Myanmar',
            'Republic of the Congo': 'Congo',
            'Ivory Coast': "Côte d'Ivoire",
            'Eswatini': 'Swaziland',
            'Taiwan': 'Taiwan, Province of China',
            # Add more mappings as needed
        }
        return special_cases.get(name, name)


def reshape_gdp_data(filepath):
    """
    Reshape GDP data from wide to long format.

    Parameters:
    - filepath: str, path to the GDP Excel file.

    Returns:
    - gdp_long: DataFrame, reshaped GDP data with columns ['country', 'year', 'gdp']
    """
    try:
        # Load the GDP data
        print(f"Loading GDP data from: {filepath}")
        # Load the exact sheet
        sheet_to_read = 'Country-Timeseries'  # Corrected sheet name
        print(f"Reading sheet: {sheet_to_read}")
        gdp_df = pd.read_excel(filepath, sheet_name=sheet_to_read)
    except Exception as e:
        print(f"Error loading GDP data: {e}")
        return None

    # Display original columns for debugging
    print("Original GDP Data Columns:")
    print(gdp_df.columns.tolist())

    # Inspect first few rows for debugging
    print("\nFirst 5 rows of GDP data:")
    print(gdp_df.head())

    # Identify the correct country and indicator columns
    possible_country_columns = ['Country Name', 'Country', 'COUNTRY', 'country']
    possible_indicator_columns = ['Indicator Name', 'Indicator', 'INDICATOR', 'indicator']

    country_col = None
    indicator_col = None

    for col in possible_country_columns:
        if col in gdp_df.columns:
            country_col = col
            break

    for col in possible_indicator_columns:
        if col in gdp_df.columns:
            indicator_col = col
            break

    if not country_col:
        print("Error: No valid country column found in GDP data.")
        return None

    if not indicator_col:
        print("Error: No valid indicator column found in GDP data.")
        return None

    # Filter rows where Indicator Name is 'GDP (current US$)'
    gdp_filtered = gdp_df[gdp_df[indicator_col].str.strip().str.lower() == 'gdp (current us$)']

    if gdp_filtered.empty:
        print("Error: No GDP data found after filtering for 'GDP (current US$)'.")
        return None

    # Reshape from wide to long format using melt
    gdp_long = gdp_filtered.melt(
        id_vars=[country_col, indicator_col],
        var_name="year",
        value_name="gdp"
    )

    # Rename columns for consistency
    gdp_long.rename(columns={
        country_col: "country",
        "year": "year"
    }, inplace=True)

    # Convert 'year' to numeric, coerce errors to NaN
    gdp_long["year"] = pd.to_numeric(gdp_long["year"], errors="coerce")

    # Drop rows with NaN in 'year' or 'gdp'
    gdp_long.dropna(subset=["year", "gdp"], inplace=True)

    # Convert 'gdp' to numeric if not already
    gdp_long["gdp"] = pd.to_numeric(gdp_long["gdp"], errors="coerce")
    gdp_long.dropna(subset=["gdp"], inplace=True)

    # Drop 'Indicator Name' as it's redundant now
    gdp_long.drop(columns=[indicator_col], inplace=True)

    # Standardize country names
    gdp_long['country'] = gdp_long['country'].apply(standardize_country_name)

    # Ensure 'year' is integer
    gdp_long['year'] = gdp_long['year'].astype(int)

    # Check for duplicates
    duplicates = gdp_long.duplicated(subset=['country', 'year']).sum()
    if duplicates > 0:
        print(f"Warning: {duplicates} duplicate records found in GDP data. Keeping the first occurrence.")
        gdp_long.drop_duplicates(subset=['country', 'year'], inplace=True)

    print("\nReshaped GDP Data:")
    print(gdp_long.head())

    return gdp_long


def load_fsi_data(folder_path):
    """
    Load and concatenate all FSI Excel files in the specified folder.

    Parameters:
    - folder_path: str, path to the folder containing FSI Excel files.

    Returns:
    - fsi_df: DataFrame, concatenated FSI data with necessary columns.
    """
    # Pattern to match FSI files, assuming filenames contain 'fsi' and end with '.xlsx'
    fsi_files_pattern = os.path.join(folder_path, "*fsi*.xlsx")
    fsi_files = glob(fsi_files_pattern)

    if not fsi_files:
        print(f"No FSI files found in {folder_path} with pattern '*fsi*.xlsx'.")
        return None

    df_list = []
    for file in fsi_files:
        try:
            print(f"Loading FSI data from: {file}")
            df = pd.read_excel(file, sheet_name=0)

            # Extract year from filename
            basename = os.path.basename(file)
            year_match = re.search(r'\d{4}', basename)
            if year_match:
                year = int(year_match.group())
            else:
                print(f"Warning: Could not extract year from filename {basename}. Skipping file.")
                continue

            # Add or overwrite the 'year' column with the extracted year
            df['year'] = year

            df_list.append(df)
        except Exception as e:
            print(f"Error loading FSI data from {file}: {e}")

    if not df_list:
        print("No FSI data loaded.")
        return None

    # Concatenate all FSI data
    fsi_df = pd.concat(df_list, ignore_index=True)

    # Display columns for debugging
    print("\nFSI Data Columns:")
    print(fsi_df.columns.tolist())

    # Inspect first few rows
    print("\nFirst 5 rows of FSI data:")
    print(fsi_df.head())

    # Rename columns for consistency
    # Adjust these mappings based on your actual FSI data column names
    # Example: 'C1' -> 'C1: Security Apparatus', etc.
    # Modify as per your FSI data
    fsi_df.rename(columns={
        "Country": "country",
        "Rank": "rank",
        "C1": "C1: Security Apparatus",
        "C2": "C2: Factionalized Elites",
        "C3": "C3: Group Grievance",
        "E1": "E1: Economy",
        "E2": "E2: Economic Inequality",
        "E3": "E3: Human Flight and Brain Drain",
        "P1": "P1: State Legitimacy",
        "P2": "P2: Public Services",
        "P3": "P3: Human Rights",
        "S1": "S1: Demographic Pressures",
        "S2": "S2: Refugees and IDPs",
        "X1": "X1: External Intervention"
        # Add more mappings as needed
    }, inplace=True)

    # Ensure FSI feature columns are present
    fsi_feature_columns = [
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
        'X1: External Intervention'
    ]

    missing_features = [col for col in fsi_feature_columns if col not in fsi_df.columns]
    if missing_features:
        print(f"Error: Missing FSI feature columns: {missing_features}")
        return None

    # Convert 'year' to numeric
    fsi_df["year"] = pd.to_numeric(fsi_df["year"], errors="coerce")
    fsi_df.dropna(subset=["year"], inplace=True)
    fsi_df["year"] = fsi_df["year"].astype(int)

    # Standardize country names
    fsi_df['country'] = fsi_df['country'].apply(standardize_country_name)

    # Drop unnecessary columns if present (e.g., 'Change from Previous Year')
    columns_to_drop = ['Change from Previous Year'] if 'Change from Previous Year' in fsi_df.columns else []
    fsi_df.drop(columns=columns_to_drop, inplace=True)

    print("\nProcessed FSI Data:")
    print(fsi_df.head())

    return fsi_df


def merge_data(gdp_long, fsi_df):
    """
    Merge GDP data with FSI data on 'country' and 'year', and create target variable 'gdp_next_year'.

    Parameters:
    - gdp_long: DataFrame, reshaped GDP data.
    - fsi_df: DataFrame, concatenated FSI data.

    Returns:
    - merged_df: DataFrame, merged dataset with features and target.
    """
    # Merge FSI with GDP on 'country' and 'year'
    print("\nMerging FSI and GDP data...")
    merged_df = pd.merge(fsi_df, gdp_long, on=['country', 'year'], how='inner')
    print(f"Number of records after initial merge: {len(merged_df)}")

    # Sort by country and year to ensure correct order
    merged_df.sort_values(by=['country', 'year'], inplace=True)

    # Create 'gdp_next_year' by shifting GDP within each country
    merged_df['gdp_next_year'] = merged_df.groupby('country')['gdp'].shift(-1)

    # Drop the last year for each country as it won't have next year's GDP
    merged_df.dropna(subset=['gdp_next_year'], inplace=True)
    print(f"Number of records after dropping rows without 'gdp_next_year': {len(merged_df)}")

    # Feature columns from FSI data and 'gdp'
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

    # Ensure all feature columns are present and handle missing values
    for col in feature_columns:
        if merged_df[col].isnull().any():
            print(f"Warning: Missing values found in feature column '{col}'. Filling with median.")
            merged_df[col].fillna(merged_df[col].median(), inplace=True)

    # Define feature matrix X and target vector y
    # Note: We're not returning X and y separately to avoid misalignment
    # Instead, we'll save the merged_df and handle feature-target separation in model_training.py
    merged_df = merged_df[['country', 'year', 'gdp_next_year'] + feature_columns]

    print("\nSample of Merged Data with Target Variable:")
    print(merged_df.head())

    return merged_df


def save_merged_data(merged_df, output_path):
    """
    Save the merged data to a CSV file.

    Parameters:
    - merged_df: DataFrame, merged dataset with features and target.
    - output_path: str, path to save the merged CSV.
    """
    try:
        merged_df.to_csv(output_path, index=False)
        print(f"\nData preparation completed. Merged data saved to '{output_path}'.")
    except Exception as e:
        print(f"Error saving merged data: {e}")


def main():
    # Step 1: Reshape GDP data
    gdp_long = reshape_gdp_data(GDP_FILE_PATH)
    if gdp_long is None:
        print("Failed to reshape GDP data.")
        return

    # Step 2: Load and combine FSI data
    fsi_df = load_fsi_data(FSI_DATA_FOLDER)
    if fsi_df is None:
        print("Failed to load FSI data.")
        return

    # Print data ranges for debugging
    print(f"GDP data years range from {gdp_long['year'].min()} to {gdp_long['year'].max()}")
    print(f"FSI data years range from {fsi_df['year'].min()} to {fsi_df['year'].max()}")

    # Check country overlap
    gdp_countries = set(gdp_long['country'].unique())
    fsi_countries = set(fsi_df['country'].unique())
    common_countries = gdp_countries.intersection(fsi_countries)
    print(f"\nNumber of unique countries in GDP data: {len(gdp_countries)}")
    print(f"Number of unique countries in FSI data: {len(fsi_countries)}")
    print(f"Number of common countries: {len(common_countries)}")

    if len(common_countries) == 0:
        print("Error: No common countries between GDP and FSI data after standardization.")
        return

    # Step 3: Merge GDP and FSI data and create target variable
    merged_df = merge_data(gdp_long, fsi_df)
    if merged_df is None or merged_df.empty:
        print("Failed to merge data.")
        return

    # Print merged data years range
    print(f"Merged data years range from {merged_df['year'].min()} to {merged_df['year'].max()}")

    # Step 4: Save the merged data to CSV
    save_merged_data(merged_df, OUTPUT_MERGED_CSV_PATH)


if __name__ == "__main__":
    main()
