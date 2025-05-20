import pandas as pd

# Завантаження даних
file_path = 'cars_dataset.csv'  # Вкажіть шлях до вашого файлу
df = pd.read_csv(file_path)

# Перевірка на пропущені значення
print("Пропущені значення:\n", df.isnull().sum())

# Видалення дублікатів
df = df.drop_duplicates()

# Уніфікація категорій (наприклад, стан авто)
df['Condition'] = df['Condition'].str.lower().replace({
    'like new': 'new',
    'used': 'old',
    'new': 'new'
})

# Приведення булевих значень до одного формату
df['Automatic'] = df['Automatic'].replace({
    'true': True, 'false': False, '+': True, '-': False
})

# Приведення числових значень до єдиного формату (наприклад, округлення цін)
df['Price'] = df['Price'].round(2)

# Збереження очищеного датасету
cleaned_file_path = 'cleaned_cars_dataset.csv'
df.to_csv(cleaned_file_path, index=False)

print(f"Очищений датасет збережено у файл: {cleaned_file_path}")