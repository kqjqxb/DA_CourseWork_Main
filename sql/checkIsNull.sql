-- Перевірка NULL у fact_car_prices
SELECT
    'fact_car_prices' AS table_name,
    COUNT(*) AS total_rows,
    SUM(CASE WHEN id IS NULL THEN 1 ELSE 0 END) AS null_id,
    SUM(CASE WHEN car_id IS NULL THEN 1 ELSE 0 END) AS null_car_id,
    SUM(CASE WHEN car_prediction2_id IS NULL THEN 1 ELSE 0 END) AS null_car_prediction2_id,
    SUM(CASE WHEN car_prediction3_id IS NULL THEN 1 ELSE 0 END) AS null_car_prediction3_id,
    SUM(CASE WHEN source_table IS NULL THEN 1 ELSE 0 END) AS null_source_table,
    SUM(CASE WHEN price IS NULL THEN 1 ELSE 0 END) AS null_price,
    SUM(CASE WHEN year_id IS NULL THEN 1 ELSE 0 END) AS null_year_id,
    SUM(CASE WHEN brand_id IS NULL THEN 1 ELSE 0 END) AS null_brand_id,
    SUM(CASE WHEN fuel_type_id IS NULL THEN 1 ELSE 0 END) AS null_fuel_type_id,
    SUM(CASE WHEN transmission_id IS NULL THEN 1 ELSE 0 END) AS null_transmission_id,
    SUM(CASE WHEN additional_id IS NULL THEN 1 ELSE 0 END) AS null_additional_id,
    SUM(CASE WHEN model IS NULL THEN 1 ELSE 0 END) AS null_model
FROM main.fact_car_prices
UNION ALL
-- Перевірка NULL у dim_brands
SELECT
    'dim_brands' AS table_name,
    COUNT(*) AS total_rows,
    SUM(CASE WHEN id IS NULL THEN 1 ELSE 0 END) AS null_id,
    SUM(CASE WHEN brand IS NULL THEN 1 ELSE 0 END) AS null_brand,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0
FROM main.dim_brands
UNION ALL
-- Перевірка NULL у dim_fuel_types
SELECT
    'dim_fuel_types' AS table_name,
    COUNT(*) AS total_rows,
    SUM(CASE WHEN id IS NULL THEN 1 ELSE 0 END) AS null_id,
    SUM(CASE WHEN fuel_type IS NULL THEN 1 ELSE 0 END) AS null_fuel_type,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0
FROM main.dim_fuel_types
UNION ALL
-- Перевірка NULL у dim_transmissions
SELECT
    'dim_transmissions' AS table_name,
    COUNT(*) AS total_rows,
    SUM(CASE WHEN id IS NULL THEN 1 ELSE 0 END) AS null_id,
    SUM(CASE WHEN transmission IS NULL THEN 1 ELSE 0 END) AS null_transmission,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0
FROM main.dim_transmissions
UNION ALL
-- Перевірка NULL у dim_year
SELECT
    'dim_year' AS table_name,
    COUNT(*) AS total_rows,
    SUM(CASE WHEN year_id IS NULL THEN 1 ELSE 0 END) AS null_year_id,
    SUM(CASE WHEN production_year IS NULL THEN 1 ELSE 0 END) AS null_production_year,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0
FROM main.dim_year
UNION ALL
-- Перевірка NULL у dim_additional_info
SELECT
    'dim_additional_info' AS table_name,
    COUNT(*) AS total_rows,
    SUM(CASE WHEN additional_id IS NULL THEN 1 ELSE 0 END) AS null_additional_id,
    SUM(CASE WHEN mileage IS NULL THEN 1 ELSE 0 END) AS null_mileage,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0
FROM main.dim_additional_info
UNION ALL
-- Перевірка NULL у fact_leather_interior
SELECT
    'fact_leather_interior' AS table_name,
    COUNT(*) AS total_rows,
    SUM(CASE WHEN id IS NULL THEN 1 ELSE 0 END) AS null_id,
    SUM(CASE WHEN car_price_id IS NULL THEN 1 ELSE 0 END) AS null_car_price_id,
    SUM(CASE WHEN leather_interior IS NULL THEN 1 ELSE 0 END) AS null_leather_interior,
    0, 0, 0, 0, 0, 0, 0, 0, 0
FROM main.fact_leather_interior
UNION ALL
-- Перевірка NULL у fact_locations
SELECT
    'fact_locations' AS table_name,
    COUNT(*) AS total_rows,
    SUM(CASE WHEN id IS NULL THEN 1 ELSE 0 END) AS null_id,
    SUM(CASE WHEN car_price_id IS NULL THEN 1 ELSE 0 END) AS null_car_price_id,
    SUM(CASE WHEN location IS NULL THEN 1 ELSE 0 END) AS null_location,
    0, 0, 0, 0, 0, 0, 0, 0, 0
FROM main.fact_locations;