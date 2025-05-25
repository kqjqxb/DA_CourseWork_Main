import pandas as pd
from sqlalchemy import create_engine

# Replace with your actual PostgreSQL connection string.
connection_string = "postgresql://postgres:09864542@localhost:5433/data_analysis"
engine = create_engine(connection_string)

# Read data from the fact_car_prices table
df = pd.read_sql("SELECT * FROM main.fact_car_prices", engine)

# Display dataframe info to check data types and non-null counts
print("DataFrame Info:")
print(df.info())

# Check for null values in each column
print("\nNon-Null Value Counts:")
print(df.notnull().sum())

# Print data types of columns
print("\nData Types:")
print(df.dtypes)


print(df['source_table'].value_counts())


