-- 1. Перевірка fact_car_prices на повністю NULL ідентифікатори
DELETE FROM main.fact_car_prices
WHERE car_id IS NULL AND car_prediction2_id IS NULL AND car_prediction3_id IS NULL;

-- 2. Виправлення dim_brands
UPDATE main.dim_brands
SET brand = 'Unknown'
WHERE brand IS NULL;

-- Видалення, якщо немає зв’язків
DELETE FROM main.dim_brands
WHERE brand = 'Unknown'
AND NOT EXISTS (
    SELECT 1 FROM main.fact_car_prices fcp WHERE fcp.brand_id = main.dim_brands.id
);

-- 3. Виправлення dim_fuel_types
UPDATE main.dim_fuel_types
SET fuel_type = 'Unknown'
WHERE fuel_type IS NULL;

-- Видалення, якщо немає зв’язків
DELETE FROM main.dim_fuel_types
WHERE fuel_type = 'Unknown'
AND NOT EXISTS (
    SELECT 1 FROM main.fact_car_prices fcp WHERE fcp.fuel_type_id = main.dim_fuel_types.id
);

-- 4. Виправлення dim_transmissions
UPDATE main.dim_transmissions
SET transmission = 'Unknown'
WHERE transmission IS NULL;

-- Видалення, якщо немає зв’язків
DELETE FROM main.dim_transmissions
WHERE transmission = 'Unknown'
AND NOT EXISTS (
    SELECT 1 FROM main.fact_car_prices fcp WHERE fcp.transmission_id = main.dim_transmissions.id
);

-- 5. Виправлення dim_year
DELETE FROM main.dim_year
WHERE production_year IS NULL
AND NOT EXISTS (
    SELECT 1 FROM main.fact_car_prices fcp WHERE fcp.year_id = main.dim_year.year_id
);

-- 6. Виправлення dim_additional_info
UPDATE main.dim_additional_info
SET mileage = 0
WHERE mileage IS NULL;

-- Перевірка результатів
SELECT
    'fact_car_prices' AS table_name,
    COUNT(*) AS total_rows,
    SUM(CASE WHEN car_id IS NULL AND car_prediction2_id IS NULL AND car_prediction3_id IS NULL THEN 1 ELSE 0 END) AS null_all_ids
FROM main.fact_car_prices
UNION ALL
SELECT
    'dim_brands' AS table_name,
    COUNT(*) AS total_rows,
    SUM(CASE WHEN brand IS NULL THEN 1 ELSE 0 END) AS null_brand
FROM main.dim_brands
UNION ALL
SELECT
    'dim_fuel_types' AS table_name,
    COUNT(*) AS total_rows,
    SUM(CASE WHEN fuel_type IS NULL THEN 1 ELSE 0 END) AS null_fuel_type
FROM main.dim_fuel_types
UNION ALL
SELECT
    'dim_transmissions' AS table_name,
    COUNT(*) AS total_rows,
    SUM(CASE WHEN transmission IS NULL THEN 1 ELSE 0 END) AS null_transmission
FROM main.dim_transmissions
UNION ALL
SELECT
    'dim_year' AS table_name,
    COUNT(*) AS total_rows,
    SUM(CASE WHEN production_year IS NULL THEN 1 ELSE 0 END) AS null_production_year
FROM main.dim_year
UNION ALL
SELECT
    'dim_additional_info' AS table_name,
    COUNT(*) AS total_rows,
    SUM(CASE WHEN mileage IS NULL THEN 1 ELSE 0 END) AS null_mileage
FROM main.dim_additional_info;