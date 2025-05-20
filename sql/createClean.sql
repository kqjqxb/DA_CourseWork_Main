-- Очищення main-зони для повторного запуску (опціонально)
TRUNCATE TABLE main.fact_locations, main.fact_leather_interior, main.fact_car_prices, main.dim_additional_info, main.dim_transmissions, main.dim_fuel_types, main.dim_brands, main.dim_year RESTART IDENTITY CASCADE;

-- Заповнення dim_brands
INSERT INTO main.dim_brands (brand)
SELECT DISTINCT brand FROM stage.car_price_prediction1
UNION
SELECT DISTINCT manufacturer FROM stage.car_price_prediction2
UNION
SELECT DISTINCT brand FROM stage.car_price_prediction3;

-- Заповнення dim_fuel_types з нормалізацією
INSERT INTO main.dim_fuel_types (fuel_type)
SELECT DISTINCT
    CASE
        WHEN fuel_type IN ('Gasolina') THEN 'Petrol'
        WHEN fuel_type IN ('Diésel') THEN 'Diesel'
        WHEN fuel_type IN ('Híbrido') THEN 'Hybrid'
        ELSE fuel_type
    END AS fuel_type
FROM (
    SELECT fuel_type FROM stage.car_price_prediction1
    UNION
    SELECT fuel_type FROM stage.car_price_prediction2
    UNION
    SELECT fuel AS fuel_type FROM stage.car_price_prediction3
) AS combined_fuel_types;

-- Заповнення dim_transmissions з нормалізацією
INSERT INTO main.dim_transmissions (transmission)
SELECT DISTINCT
    CASE
        WHEN gearbox_type = 'Automatica' THEN 'Automatic'
        ELSE gearbox_type
    END AS transmission
FROM (
    SELECT gearbox_type FROM stage.car_price_prediction1
    UNION
    SELECT gearbox_type FROM stage.car_price_prediction2
    UNION
    SELECT gearbox_type FROM stage.car_price_prediction3
) AS combined_gearbox_types;

-- Заповнення dim_year
INSERT INTO main.dim_year (production_year)
SELECT DISTINCT year FROM stage.car_price_prediction1
UNION
SELECT DISTINCT prod_year FROM stage.car_price_prediction2
UNION
SELECT DISTINCT year FROM stage.car_price_prediction3;

-- Заповнення dim_additional_info (пробіг)
INSERT INTO main.dim_additional_info (mileage)
SELECT mileage FROM stage.car_price_prediction1
UNION
SELECT mileage FROM stage.car_price_prediction2
UNION
SELECT mileage FROM stage.car_price_prediction3;

-- Заповнення fact_car_prices з car_price_prediction1
INSERT INTO main.fact_car_prices (
    car_id,
    source_table,
    price,
    year_id,
    brand_id,
    fuel_type_id,
    transmission_id,
    additional_id,
    model
)
SELECT
    p1.car_id,
    'prediction1' AS source_table,
    p1.price,
    y.year_id,
    b.id AS brand_id,
    f.id AS fuel_type_id,
    t.id AS transmission_id,
    a.additional_id,
    p1.model
FROM stage.car_price_prediction1 p1
JOIN main.dim_year y ON p1.year = y.production_year
JOIN main.dim_brands b ON p1.brand = b.brand
JOIN main.dim_fuel_types f ON p1.fuel_type = f.fuel_type
JOIN main.dim_transmissions t ON p1.gearbox_type = t.transmission
JOIN main.dim_additional_info a ON p1.mileage = a.mileage;

-- Заповнення fact_car_prices з car_price_prediction2
INSERT INTO main.fact_car_prices (
    car_prediction2_id,
    source_table,
    price,
    year_id,
    brand_id,
    fuel_type_id,
    transmission_id,
    additional_id,
    model
)
SELECT
    p2.id,
    'prediction2' AS source_table,
    p2.price,
    y.year_id,
    b.id AS brand_id,
    f.id AS fuel_type_id,
    t.id AS transmission_id,
    a.additional_id,
    p2.model
FROM stage.car_price_prediction2 p2
JOIN main.dim_year y ON p2.prod_year = y.production_year
JOIN main.dim_brands b ON p2.manufacturer = b.brand
JOIN main.dim_fuel_types f ON p2.fuel_type = f.fuel_type
JOIN main.dim_transmissions t ON p2.gearbox_type = t.transmission
JOIN main.dim_additional_info a ON p2.mileage = a.mileage;

-- Заповнення fact_car_prices з car_price_prediction3
INSERT INTO main.fact_car_prices (
    car_prediction3_id,
    source_table,
    price,
    year_id,
    brand_id,
    fuel_type_id,
    transmission_id,
    additional_id,
    model
)
SELECT
    p3.id,
    'prediction3' AS source_table,
    p3.price,
    y.year_id,
    b.id AS brand_id,
    f.id AS fuel_type_id,
    t.id AS transmission_id,
    a.additional_id,
    p3.model
FROM stage.car_price_prediction3 p3
JOIN main.dim_year y ON p3.year = y.production_year
JOIN main.dim_brands b ON p3.brand = b.brand
JOIN main.dim_fuel_types f ON (
    CASE
        WHEN p3.fuel = 'Gasolina' THEN 'Petrol'
        WHEN p3.fuel = 'Diésel' THEN 'Diesel'
        WHEN p3.fuel = 'Híbrido' THEN 'Hybrid'
        ELSE p3.fuel
    END
) = f.fuel_type
JOIN main.dim_transmissions t ON (
    CASE
        WHEN p3.gearbox_type = 'Automatica' THEN 'Automatic'
        ELSE p3.gearbox_type
    END
) = t.transmission
JOIN main.dim_additional_info a ON p3.mileage = a.mileage;

-- Заповнення fact_leather_interior
INSERT INTO main.fact_leather_interior (car_price_id, leather_interior)
SELECT
    fcp.id,
    p2.leather_interior
FROM main.fact_car_prices fcp
JOIN stage.car_price_prediction2 p2 ON fcp.car_prediction2_id = p2.id
WHERE fcp.source_table = 'prediction2';

-- Заповнення fact_locations
INSERT INTO main.fact_locations (car_price_id, location)
SELECT
    fcp.id,
    p3.location
FROM main.fact_car_prices fcp
JOIN stage.car_price_prediction3 p3 ON fcp.car_prediction3_id = p3.id
WHERE fcp.source_table = 'prediction3';