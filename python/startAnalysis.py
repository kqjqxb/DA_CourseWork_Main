# startAnalysis.py
# Script for car price prediction and market analysis using regression models
# Generates model comparisons, top car lists, and visualizations

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from db_connection import sqlalchemy_engine  # Assumes you have a module for DB connection

# Set random seed for reproducibility
np.random.seed(42)

# 1. Load data from PostgreSQL
print("Loading data from database...")
query = """
        SELECT f.price, \
               b.brand, \
               f.model, \
               y.production_year, \
               a.mileage, \
               ft.fuel_type, \
               t.transmission, \
               li.leather_interior, \
               l.location, \
               f.source_table
        FROM main.fact_car_prices f
                 JOIN main.dim_brands b ON f.brand_id = b.id
                 JOIN main.dim_year y ON f.year_id = y.year_id
                 JOIN main.dim_additional_info a ON f.additional_id = a.additional_id
                 JOIN main.dim_fuel_types ft ON f.fuel_type_id = ft.id
                 JOIN main.dim_transmissions t ON f.transmission_id = t.id
                 LEFT JOIN main.fact_leather_interior li ON f.id = li.car_price_id
                 LEFT JOIN main.fact_locations l ON f.id = l.car_price_id; \
        """
df = pd.read_sql(query, sqlalchemy_engine)

# Print first 5 rows and data info for debugging
print("First 5 rows of loaded data:")
print(df.head())
print("\nData info:")
print(df.info())

# 2. Data cleaning
print("Cleaning data...")
# Handle missing values
df['mileage'] = df['mileage'].fillna(df['mileage'].mean())
df['leather_interior'] = df['leather_interior'].astype(bool).fillna(False)  # Explicitly cast to bool
df['location'] = df['location'].fillna(df['location'].mode()[0])
df['fuel_type'] = df['fuel_type'].fillna(df['fuel_type'].mode()[0])
df['transmission'] = df['transmission'].fillna(df['transmission'].mode()[0])
df['model'] = df['model'].fillna(df['model'].mode()[0])
df['price'] = df['price'].fillna(df['price'].mean())  # Handle missing prices

# Remove outliers
df = df[(df['price'] > 100) & (df['price'] < 100000)]  # Reasonable price range
df = df[df['mileage'] < 1000000]  # Reasonable mileage
df = df[df['production_year'] > 1980]  # Reasonable year

# Convert prices to USD based on source_table
print("Converting prices to USD...")
df.loc[df['source_table'] == 'prediction3', 'price'] = df['price'] * 1.1  # EUR to USD for prediction3

# 3. Feature encoding for training
print("Encoding features for training...")
# One-hot encode categorical features
categorical_cols = ['brand', 'fuel_type', 'transmission', 'location']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Limit models to top-50 to reduce dimensionality for training
top_models = df['model'].value_counts().head(50).index
df_encoded = df_encoded[df_encoded['model'].isin(top_models)]
df_encoded['model'] = df_encoded['model'].astype('category').cat.codes

# Convert leather_interior to binary
df_encoded['leather_interior'] = df_encoded['leather_interior'].astype(int)

# Drop source_table as it's not needed for modeling
df_encoded = df_encoded.drop(columns=['source_table'])

# 4. Prepare features and target for training
print("Preparing features and target for training...")
X = df_encoded.drop(columns=['price'])
y = df_encoded['price']

# Standardize numerical features
scaler = StandardScaler()
numerical_cols = ['production_year', 'mileage', 'model']
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Model training and evaluation
print("Training models...")
models = {
    'Random Forest': RandomForestRegressor(random_state=42),
    'KNN': KNeighborsRegressor(),
    'Linear Regression': LinearRegression()
}

# Grid search parameters
param_grids = {
    'Random Forest': {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    },
    'KNN': {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance']
    },
    'Linear Regression': {}  # No parameters to tune
}

# Store results
results = []

for name, model in models.items():
    print(f"Training {name}...")
    if param_grids[name]:
        grid = GridSearchCV(model, param_grids[name], cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        print(f"Best parameters for {name}: {grid.best_params_}")
    else:
        best_model = model
        best_model.fit(X_train, y_train)

    # Predictions
    y_pred = best_model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({
        'Model': name,
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2
    })

    # Feature importance for Random Forest
    if name == 'Random Forest':
        importance = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importance.values, y=importance.index)
        plt.title('Feature Importance (Random Forest)')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()

# 6. Compare models
print("\nModel Comparison:")
results_df = pd.DataFrame(results)
print(results_df)

# Save results to CSV
results_df.to_csv('model_comparison.csv', index=False)
print("Model comparison saved to 'model_comparison.csv'")

# 7. Predict prices for top lists (using Random Forest)
print("Predicting prices for top lists...")
# Create full feature set for prediction (without filtering models)
df_full_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
df_full_encoded['model'] = df_full_encoded['model'].astype('category').cat.codes
df_full_encoded['leather_interior'] = df_full_encoded['leather_interior'].astype(int)
df_full_encoded = df_full_encoded.drop(columns=['source_table'])

# Ensure columns match X (used for training)
missing_cols = [col for col in X.columns if col not in df_full_encoded.columns]
for col in missing_cols:
    df_full_encoded[col] = 0
df_full_encoded = df_full_encoded[X.columns]  # Reorder columns to match X

# Standardize numerical features for full dataset
df_full_encoded[numerical_cols] = scaler.transform(df_full_encoded[numerical_cols])

# Predict prices
rf_model = models['Random Forest'].fit(X_train, y_train)
df['predicted_price'] = rf_model.predict(df_full_encoded)

# 8. Create top lists
print("Creating top lists...")
# Budget Top (price < 10,000 USD, year > 2010, mileage < 150,000 km)
budget_top = df[
    (df['price'] < 10000) &
    (df['production_year'] > 2010) &
    (df['mileage'] < 150000)
    ][['brand', 'model', 'production_year', 'mileage', 'price', 'fuel_type']].head(5)

# Premium Top (price > 30,000 USD, year > 2018, leather_interior = True)
premium_top = df[
    (df['price'] > 30000) &
    (df['production_year'] > 2018) &
    (df['leather_interior'] == True)
    ][['brand', 'model', 'production_year', 'mileage', 'price', 'fuel_type']].head(5)

# Economic Top (Hybrid/Electric, mileage < 100,000 km)
economic_top = df[
    (df['fuel_type'].isin(['Hybrid', 'Electric'])) &
    (df['mileage'] < 100000)
    ][['brand', 'model', 'production_year', 'mileage', 'price', 'fuel_type']].head(5)

# Price/Quality Top (largest negative difference: price - predicted_price)
df['price_diff'] = df['price'] - df['predicted_price']
price_quality_top = df.sort_values('price_diff')[
    ['brand', 'model', 'production_year', 'mileage', 'price', 'fuel_type']].head(5)

# Save tops to CSV
budget_top.to_csv('budget_top.csv', index=False)
premium_top.to_csv('premium_top.csv', index=False)
economic_top.to_csv('economic_top.csv', index=False)
price_quality_top.to_csv('price_quality_top.csv', index=False)

# 9. Visualizations
print("Generating visualizations...")
# Predicted vs Actual Prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, rf_model.predict(X_test), alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Price (USD)')
plt.ylabel('Predicted Price (USD)')
plt.title('Predicted vs Actual Prices (Random Forest)')
plt.tight_layout()
plt.savefig('predicted_vs_actual.png')
plt.close()

# Price distribution by fuel type
plt.figure(figsize=(10, 6))
sns.boxplot(x='fuel_type', y='price', data=df)
plt.title('Price Distribution by Fuel Type')
plt.xlabel('Fuel Type')
plt.ylabel('Price (USD)')
plt.tight_layout()
plt.savefig('price_by_fuel_type.png')
plt.close()

# 10. Print top lists
print("\nTop Lists:")
print("\nBudget Top (Price < $10,000, Year > 2010, Mileage < 150,000 km):")
print(budget_top)
print("\nPremium Top (Price > $30,000, Year > 2018, Leather Interior):")
print(premium_top)
print("\nEconomic Top (Hybrid/Electric, Mileage < 100,000 km):")
print(economic_top)
print("\nPrice/Quality Top (Best value based on predicted price):")
print(price_quality_top)

print("\nAll results and visualizations saved successfully!")