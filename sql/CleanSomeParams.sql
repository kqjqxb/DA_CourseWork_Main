-- 1. Нормалізація dim_brands (приведення до формату з першою великою літерою)
-- Створюємо тимчасову таблицю для нормалізованих брендів
CREATE TEMP TABLE temp_brands AS
SELECT
    MIN(id) AS id,
    INITCAP(brand) AS normalized_brand
FROM main.dim_brands
GROUP BY INITCAP(brand);

-- Оновлюємо fact_car_prices, щоб використовувати найменший id для нормалізованих брендів
UPDATE main.fact_car_prices fcp
SET brand_id = tb.id
FROM main.dim_brands db
JOIN temp_brands tb ON INITCAP(db.brand) = tb.normalized_brand
WHERE fcp.brand_id = db.id;

-- Оновлюємо dim_brands, щоб зберегти нормалізовані назви
UPDATE main.dim_brands db
SET brand = tb.normalized_brand
FROM temp_brands tb
WHERE db.id = tb.id;

-- Видаляємо дублікати в dim_brands
DELETE FROM main.dim_brands
WHERE id NOT IN (SELECT id FROM temp_brands);

-- Очищаємо тимчасову таблицю
DROP TABLE temp_brands;

-- 2. Нормалізація dim_fuel_types
-- Створюємо тимчасову таблицю для нормалізованих типів пального
CREATE TEMP TABLE temp_fuel_types AS
SELECT
    MIN(id) AS id,
    CASE
        WHEN fuel_type IN ('CNG', 'GLP', 'LPG') THEN 'Gas'
        WHEN fuel_type IN ('Eléctrico', 'Electric') THEN 'Electric'
        WHEN fuel_type IN ('Plug-in Hybrid', 'Hybrid') THEN 'Hybrid'
        ELSE fuel_type
    END AS normalized_fuel_type
FROM main.dim_fuel_types
GROUP BY
    CASE
        WHEN fuel_type IN ('CNG', 'GLP', 'LPG') THEN 'Gas'
        WHEN fuel_type IN ('Eléctrico', 'Electric') THEN 'Electric'
        WHEN fuel_type IN ('Plug-in Hybrid', 'Hybrid') THEN 'Hybrid'
        ELSE fuel_type
    END;

-- Оновлюємо fact_car_prices, щоб використовувати найменший id для нормалізованих типів пального
UPDATE main.fact_car_prices fcp
SET fuel_type_id = tft.id
FROM main.dim_fuel_types dft
JOIN temp_fuel_types tft ON (
    CASE
        WHEN dft.fuel_type IN ('CNG', 'GLP', 'LPG') THEN 'Gas'
        WHEN dft.fuel_type IN ('Eléctrico', 'Electric') THEN 'Electric'
        WHEN dft.fuel_type IN ('Plug-in Hybrid', 'Hybrid') THEN 'Hybrid'
        ELSE dft.fuel_type
    END
) = tft.normalized_fuel_type
WHERE fcp.fuel_type_id = dft.id;

-- Оновлюємо dim_fuel_types, щоб зберегти нормалізовані назви
UPDATE main.dim_fuel_types dft
SET fuel_type = tft.normalized_fuel_type
FROM temp_fuel_types tft
WHERE dft.id = tft.id;

-- Видаляємо дублікати в dim_fuel_types
DELETE FROM main.dim_fuel_types
WHERE id NOT IN (SELECT id FROM temp_fuel_types);

-- Очищаємо тимчасову таблицю
DROP TABLE temp_fuel_types;

-- Перевірка результатів
SELECT * FROM main.dim_brands ORDER BY brand;
SELECT * FROM main.dim_fuel_types ORDER BY fuel_type;