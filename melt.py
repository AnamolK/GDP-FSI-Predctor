import pandas as pd

# File path
GDP_FILE_PATH = r"C:\Users\kaspa\PycharmProjects\GDP\data\gdp\WITS-Country-Timeseries.xlsx"

def reshape_gdp_data(filepath):
    """
    Reshape the GDP data from wide to long format.
    """
    # Load the data
    gdp_df = pd.read_excel(filepath)

    # Reshape from wide to long format
    gdp_long = gdp_df.melt(
        id_vars=["Country Name", "Indicator Name"],
        var_name="Year",
        value_name="GDP (current US$)"
    )

    # Rename columns for consistency with the FSI data
    gdp_long.rename(columns={"Country Name": "country", "Year": "year", "GDP (current US$)": "gdp"}, inplace=True)

    # Convert year to integer and drop rows without GDP data
    gdp_long["year"] = pd.to_numeric(gdp_long["year"], errors="coerce")
    gdp_long.dropna(subset=["year", "gdp"], inplace=True)

    # Drop the "Indicator Name" column if it's irrelevant for merging
    gdp_long.drop(columns=["Indicator Name"], inplace=True)

    return gdp_long

def load_gdp_data(filepath):
    """
    Load and reshape the GDP data.
    """
    return reshape_gdp_data(filepath)

def save_reshaped_data(df, output_path):
    """
    Save the reshaped data to a CSV file for inspection or later use.
    """
    df.to_csv(output_path, index=False)
    print(f"Reshaped GDP data saved to {output_path}")

# Example usage
if __name__ == "__main__":
    reshaped_gdp = load_gdp_data(GDP_FILE_PATH)
    output_file = r"C:\Users\kaspa\PycharmProjects\GDP\data\gdp\reshaped_gdp.csv"
    save_reshaped_data(reshaped_gdp, output_file)
