import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2
from sqlalchemy import create_engine
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Налаштування стилю графіків
sns.set_palette('husl')

# Підключення до бази даних
db_params = {
    'dbname': 'data_analysis',
    'user': 'postgres',
    'password': '09864542',
    'host': 'localhost',
    'port': '5433'
}

# Створення SQLAlchemy engine
engine = create_engine(f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}")

# SQL-запит для об’єднання всіх необхідних даних
query = """
SELECT 
    fcp.id,
    fcp.price,
    fcp.source_table,
    fcp.model,
    b.brand,
    y.production_year,
    ft.fuel_type,
    t.transmission,
    ai.mileage,
    li.leather_interior,
    l.location,
    c.condition,
    aa.levy,
    aa.cylinders,
    aa.drive_wheels,
    aa.doors,
    aa.wheel,
    aa.color,
    aa.airbags
FROM main.fact_car_prices fcp
LEFT JOIN main.dim_brands b ON fcp.brand_id = b.id
LEFT JOIN main.dim_year y ON fcp.year_id = y.production_year
LEFT JOIN main.dim_fuel_types ft ON fcp.fuel_type_id = ft.id
LEFT JOIN main.dim_transmissions t ON fcp.transmission_id = t.id
LEFT JOIN main.dim_additional_info ai ON fcp.additional_id = ai.additional_id
LEFT JOIN main.fact_leather_interior li ON fcp.id = li.car_price_id
LEFT JOIN main.fact_locations l ON fcp.id = l.car_price_id
LEFT JOIN main.dim_additional_attributes aa ON fcp.id = aa.car_price_id
LEFT JOIN main.dim_conditions c ON aa.condition_id = c.id
"""

# Завантаження даних у DataFrame
try:
    df = pd.read_sql(query, engine)
except Exception as e:
    print(f"Помилка підключення до бази даних або виконання запиту: {e}")
    exit(1)

# Перевірка, чи DataFrame порожній
if df.empty:
    print("Помилка: Набір даних порожній. Перевірте, чи таблиці в базі даних містять дані.")
    print("Виконайте наступні SQL-запити для діагностики:")
    print("SELECT COUNT(*) FROM main.fact_car_prices;")
    print("SELECT COUNT(*) FROM main.dim_brands;")
    print("SELECT COUNT(*) FROM main.dim_year;")
    print("SELECT COUNT(*) FROM main.dim_fuel_types;")
    print("SELECT COUNT(*) FROM main.dim_transmissions;")
    exit(1)

# Діагностика: Виведення кількості рядків і стовпців
print(f"Завантажено {df.shape[0]} рядків і {df.shape[1]} стовпців")

# Попередня обробка даних
# Перевірка пропущених значень
print("Пропущені значення:\n", df.isnull().sum())

# Заповнення пропущених значень (приклад)
df['mileage'] = df['mileage'].fillna(df['mileage'].median())
df['leather_interior'] = df['leather_interior'].fillna(False)
df['condition'] = df['condition'].fillna('Unknown')
df['location'] = df['location'].fillna('Unknown')

# Конвертація типів даних (якщо потрібно)
df['production_year'] = df['production_year'].astype(int)
df['price'] = df['price'].astype(float)
df['mileage'] = df['mileage'].astype(float)

# 1. Описовий аналіз
print("\nОписовий аналіз цін:")
print(df['price'].describe())

# Середня ціна за маркою
avg_price_by_brand = df.groupby('brand')['price'].mean().sort_values(ascending=False)
print("\nСередня ціна за маркою:\n", avg_price_by_brand.head(10))

# Середня ціна за типом пального
avg_price_by_fuel = df.groupby('fuel_type')['price'].mean().sort_values(ascending=False)
print("\nСередня ціна за типом пального:\n", avg_price_by_fuel)

# 2. Кореляційний аналіз
numeric_cols = ['price', 'production_year', 'mileage', 'cylinders', 'doors', 'airbags']
corr_matrix = df[numeric_cols].corr()
print("\nМатриця кореляцій:\n", corr_matrix)

# Візуалізація кореляційної матриці
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Кореляція числових характеристик')
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Аналіз категорій
# Boxplot цін за марками (топ-10 марок)
top_brands = df['brand'].value_counts().head(10).index
plt.figure(figsize=(12, 6))
sns.boxplot(x='brand', y='price', data=df[df['brand'].isin(top_brands)])
plt.xticks(rotation=45)
plt.title('Розподіл цін за марками (Топ-10)')
plt.savefig('price_by_brand.png', dpi=300, bbox_inches='tight')
plt.show()

# Boxplot цін за типом пального
plt.figure(figsize=(10, 6))
sns.boxplot(x='fuel_type', y='price', data=df)
plt.title('Розподіл цін за типом пального')
plt.savefig('price_by_fuel_type.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Ринкові тенденції
# Зміна середньої ціни за роками
avg_price_by_year = df.groupby('production_year')['price'].mean()
plt.figure(figsize=(12, 6))
avg_price_by_year.plot(kind='line', marker='o')
plt.title('Зміна середньої ціни за роками випуску')
plt.xlabel('Рік випуску')
plt.ylabel('Середня ціна (USD)')
plt.grid(True)
plt.savefig('price_by_year.png', dpi=300, bbox_inches='tight')
plt.show()

# Популярність марок
brand_counts = df['brand'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=brand_counts.values, y=brand_counts.index)
plt.title('Топ-10 найпопулярніших марок')
plt.xlabel('Кількість автомобілів')
plt.savefig('popular_brands.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. Регіональні тенденції
avg_price_by_location = df.groupby('location')['price'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=avg_price_by_location.values, y=avg_price_by_location.index)
plt.title('Середня ціна за локацією (Топ-10)')
plt.xlabel('Середня ціна (USD)')
plt.savefig('price_by_location.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. Регресійний аналіз (прогнозування ціни)
# Вибір ознак
features = ['production_year', 'mileage', 'cylinders', 'doors', 'airbags']
X = df[features].fillna(0)  # Заповнення пропущених значень
y = df['price']

# Розбиття даних
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Навчання моделі
model = LinearRegression()
model.fit(X_train, y_train)

# Прогнозування
y_pred = model.predict(X_test)

# Оцінка моделі
r2 = r2_score(y_test, y_pred)
print(f"\nКоефіцієнт детермінації (R²): {r2:.2f}")

# Коефіцієнти регресії
coef_df = pd.DataFrame({'Feature': features, 'Coefficient': model.coef_})
print("\nКоефіцієнти регресії:\n", coef_df)

# Візуалізація прогнозів
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Фактична ціна')
plt.ylabel('Прогнозована ціна')
plt.title('Фактичні vs Прогнозовані ціни')
plt.savefig('predicted_vs_actual_prices.png', dpi=300, bbox_inches='tight')
plt.show()