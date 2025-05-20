import pandas as pd

# Provide the absolute path to the CSV file
file_path = '/Users/mmaksim/PycharmProjects/AD_2/datasets/car_price_prediction2_cleaned.csv'

# Load the CSV file
df = pd.read_csv(file_path)

# Check if the 'Doors' column exists before processing
if 'Doors' in df.columns:
    # Fix the 'Doors' column: Replace invalid values with NaN
    df['Doors'] = pd.to_numeric(df['Doors'], errors='coerce')
else:
    print("'Doors' column not found in the dataset.")

# Fix the 'Engine volume' column: Extract numeric part and convert to float
if 'Engine volume' in df.columns:
    df['Engine volume'] = df['Engine volume'].str.extract(r'(\d+\.?\d*)').astype(float)
else:
    print("'Engine volume' column not found in the dataset.")

# Save the cleaned data to a new file
cleaned_file_path = '/Users/mmaksim/PycharmProjects/AD_2/datasets/car_price_prediction2_cleaned.csv'
df.to_csv(cleaned_file_path, index=False)

print(f"Cleaned data saved to {cleaned_file_path}")