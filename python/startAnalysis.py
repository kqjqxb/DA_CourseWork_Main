import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from db_connection import sqlalchemy_engine

# here i set random seed for reproducibility
np.random.seed(42)

# 1. here tte stem where i load data from my db
print("Loading data from database...")
query = """
SELECT
    f.price,
    b.brand,
    f.model,
    y.production_year,
    a.mileage,
    ft.fuel_type,
    t.transmission,
    li.leather_interior,
    l.location,
    f.source_table
FROM main.fact_car_prices f
JOIN main.dim_brands b ON f.brand_id = b.id
JOIN main.dim_year y ON f.year_id = y.year_id
JOIN main.dim_additional_info a ON f.additional_id = a.additional_id
JOIN main.dim_fuel_types ft ON f.fuel_type_id = ft.id
JOIN main.dim_transmissions t ON f.transmission_id = t.id
LEFT JOIN main.fact_leather_interior li ON f.id = li.car_price_id
LEFT JOIN main.fact_locations l ON f.id = l.car_price_id;
"""
df = pd.read_sql(query, sqlalchemy_engine)

# for print first 5 rows and data info for debugging
print("First 5 rows of loaded data:")
print(df.head())
print("\nData info:")
print(df.info())

# 2. here i am cleaning my data
print("Cleaning data...")

# handle missing values
df['mileage'] = df['mileage'].fillna(df['mileage'].mean())
df['leather_interior'] = df['leather_interior'].astype(bool).fillna(False)
df['location'] = df['location'].fillna(df['location'].mode()[0])
df['fuel_type'] = df['fuel_type'].fillna(df['fuel_type'].mode()[0])
df['transmission'] = df['transmission'].fillna(df['transmission'].mode()[0])
df['model'] = df['model'].fillna(df['model'].mode()[0])
df['price'] = df['price'].fillna(df['price'].mean())

# for remove outliers of my data
df = df[(df['price'] > 100) & (df['price'] < 100000)]
df = df[df['mileage'] < 1000000]
df = df[df['production_year'] > 1980]

# i converted prices to USD by multiple price for 1.1 based on source_table
print("Converting prices to USD...")
df.loc[df['source_table'] == 'prediction3', 'price'] = df['price'] * 1.1

# 3. here feature encoding for training
print("Encoding features for training...")

# encode categorical features
categorical_cols = ['brand', 'fuel_type', 'transmission', 'location']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# limit models to top-50 to reduce dimensionality for training
top_models = df['model'].value_counts().head(50).index
df_encoded = df_encoded[df_encoded['model'].isin(top_models)]
df_encoded['model'] = df_encoded['model'].astype('category').cat.codes

# convert leather_interior to binary
df_encoded['leather_interior'] = df_encoded['leather_interior'].astype(int)

# th next i drop source_table as it's not needed for modeling
df_encoded = df_encoded.drop(columns=['source_table'])

# 4. here preparing features and target for training
print("Preparing features and target for training...")
X = df_encoded.drop(columns=['price'])
y = df_encoded['price']

# standardize numerical features
scaler = StandardScaler()
numerical_cols = ['production_year', 'mileage', 'model']
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4.1. prepare data for classification (discretize prices)
print("Preparing data for classification...")

# discretize prices into categories: low (<10,000), medium (10,000-30,000), high (>30,000)
bins = [0, 10000, 30000, float('inf')]
labels = ['Low', 'Medium', 'High']
y_class = pd.cut(y, bins=bins, labels=labels, include_lowest=True)
y_train_class = pd.cut(y_train, bins=bins, labels=labels, include_lowest=True)
y_test_class = pd.cut(y_test, bins=bins, labels=labels, include_lowest=True)

# 5. model training and evaluation (Regression)
print("Training regression models...")
models = {
    'Random Forest': RandomForestRegressor(random_state=42),
    'KNN': KNeighborsRegressor(),
    'Linear Regression': LinearRegression()
}

# grid search parameters
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
    'Linear Regression': {}
}

# store results
results = []

for name, model in models.items():
    print(f"Training {name}...")
    if param_grids[name]:
        grid = GridSearchCV(model, param_grids[name], cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        print(f"Best parameters for {name}: {grid.best_params_}")
        models[name] = best_model  # update models dictionary with best model
    else:
        model.fit(X_train, y_train)
        models[name] = model  # ensure that model is updated

    # predictions on test set
    y_pred = models[name].predict(X_test)

    # metrics on test set
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # doing cross-validation
    print(f"Performing cross-validation for {name}...")
    cv_mae = -cross_val_score(models[name], X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1).mean()
    cv_rmse = np.sqrt(-cross_val_score(models[name], X, y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1).mean())
    cv_r2 = cross_val_score(models[name], X, y, cv=5, scoring='r2', n_jobs=-1).mean()

    results.append({
        'Model': name,
        'Test MAE': mae,
        'Test RMSE': rmse,
        'Test R²': r2,
        'CV MAE': cv_mae,
        'CV RMSE': cv_rmse,
        'CV R²': cv_r2
    })

    # feature importance for Random Forest
    if name == 'Random Forest':
        importance = pd.Series(models[name].feature_importances_, index=X.columns).sort_values(ascending=False)
        selected_features = [col for col in importance.index if not col.startswith('brand_') and col != 'model']
        filtered_importance = importance[selected_features]
        plt.figure(figsize=(10, 6))
        sns.barplot(x=filtered_importance.values, y=filtered_importance.index)
        plt.title('Feature Importance (Random Forest)')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()

# 5.1. KNN classification
print("Training KNN classifier...")
knn_classifier = KNeighborsClassifier()
param_grid_knn = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance']
}
grid_knn = GridSearchCV(knn_classifier, param_grid_knn, cv=5, scoring='accuracy', n_jobs=-1)
grid_knn.fit(X_train, y_train_class)
best_knn_classifier = grid_knn.best_estimator_
print(f"Best parameters for KNN Classifier: {grid_knn.best_params_}")

# predictions and confusion matrix for KNN
y_pred_class = best_knn_classifier.predict(X_test)
cm = confusion_matrix(y_test_class, y_pred_class, labels=labels)
accuracy = accuracy_score(y_test_class, y_pred_class)

# plot confusion matrix for KNN
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title(f'KNN Confusion Matrix (Accuracy: {accuracy:.2f})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('knn_confusion_matrix.png')
plt.close()

# 5.2. Random Forest classification
print("Training Random Forest classifier...")
rf_classifier = RandomForestClassifier(random_state=42)
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
grid_rf = GridSearchCV(rf_classifier, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
grid_rf.fit(X_train, y_train_class)
best_rf_classifier = grid_rf.best_estimator_
print(f"Best parameters for Random Forest Classifier: {grid_rf.best_params_}")

# predictions and confusion matrix for Random Forest
y_pred_class_rf = best_rf_classifier.predict(X_test)
cm_rf = confusion_matrix(y_test_class, y_pred_class_rf, labels=labels)
accuracy_rf = accuracy_score(y_test_class, y_pred_class_rf)

# plot confusion matrix for Random Forest
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title(f'Random Forest Confusion Matrix (Accuracy: {accuracy_rf:.2f})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('rf_confusion_matrix.png')
plt.close()

# 6. compare regression models
print("\nModel Comparison:")
results_df = pd.DataFrame(results)
print(results_df)

# now i can save results to CSV
results_df.to_csv('model_comparison.csv', index=False)
print("Model comparison saved to 'model_comparison.csv'")

# 7. predict prices for top lists (using Random Forest)
print("Predicting prices for top lists...")
df_full_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
df_full_encoded['model'] = df_full_encoded['model'].astype('category').cat.codes
df_full_encoded['leather_interior'] = df_full_encoded['leather_interior'].astype(int)
df_full_encoded = df_full_encoded.drop(columns=['source_table'])

#eEnsure columns match X
missing_cols = [col for col in X.columns if col not in df_full_encoded.columns]
for col in missing_cols:
    df_full_encoded[col] = 0
df_full_encoded = df_full_encoded[X.columns]

# standardize numerical features
df_full_encoded[numerical_cols] = scaler.transform(df_full_encoded[numerical_cols])

# predict prices
rf_model = models['Random Forest']
df['predicted_price'] = rf_model.predict(df_full_encoded)

# 8. create top lists
print("Creating top lists...")
budget_top = df[
    (df['price'] < 10000) &
    (df['production_year'] > 2010) &
    (df['mileage'] < 150000)
][['brand', 'model', 'production_year', 'mileage', 'price', 'fuel_type']].head(5)

premium_top = df[
    (df['price'] > 30000) &
    (df['production_year'] > 2018) &
    (df['leather_interior'] == True)
][['brand', 'model', 'production_year', 'mileage', 'price', 'fuel_type']].head(5)

economic_top = df[
    (df['fuel_type'].isin(['Hybrid', 'Electric'])) &
    (df['mileage'] < 100000)
][['brand', 'model', 'production_year', 'mileage', 'price', 'fuel_type']].head(5)

df['price_diff'] = df['price'] - df['predicted_price']
price_quality_top = df.sort_values('price_diff')[['brand', 'model', 'production_year', 'mileage', 'price', 'fuel_type']].head(5)

# save tops to CSV
budget_top.to_csv('budget_top.csv', index=False)
premium_top.to_csv('premium_top.csv', index=False)
economic_top.to_csv('economic_top.csv', index=False)
price_quality_top.to_csv('price_quality_top.csv', index=False)

# 9. visualizations
print("Generating visualizations...")

# predicted vs actual prices (Random Forest)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, rf_model.predict(X_test), alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Price (USD)')
plt.ylabel('Predicted Price (USD)')
plt.title('Predicted vs Actual Prices (Random Forest)')
plt.tight_layout()
plt.savefig('predicted_vs_actual.png')
plt.close()

# price distribution by fuel type
plt.figure(figsize=(10, 6))
sns.boxplot(x='fuel_type', y='price', data=df)
plt.title('Price Distribution by Fuel Type')
plt.xlabel('Fuel Type')
plt.ylabel('Price (USD)')
plt.tight_layout()
plt.savefig('price_by_fuel_type.png')
plt.close()

# comparison of predicted vs actual prices for all models
plt.figure(figsize=(12, 4))

# random Forest
plt.subplot(1, 3, 1)
plt.scatter(y_test, models['Random Forest'].predict(X_test), alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Price (USD)')
plt.ylabel('Predicted Price (USD)')
plt.title('Random Forest')

# KNN
plt.subplot(1, 3, 2)
plt.scatter(y_test, models['KNN'].predict(X_test), alpha=0.5, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Price (USD)')
plt.ylabel('Predicted Price (USD)')
plt.title('KNN')

# Linear Regression
plt.subplot(1, 3, 3)
plt.scatter(y_test, models['Linear Regression'].predict(X_test), alpha=0.5, color='orange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Price (USD)')
plt.ylabel('Predicted Price (USD)')
plt.title('Linear Regression')

plt.tight_layout()
plt.savefig('all_models_predicted_vs_actual.png')
plt.close()

# 10. and at the end print top lists
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