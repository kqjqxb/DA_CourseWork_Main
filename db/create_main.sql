-- Створення main зони
CREATE SCHEMA IF NOT EXISTS main;

-- Таблиця брендів
CREATE TABLE main.dim_brands (
    id SERIAL PRIMARY KEY,
    brand VARCHAR(50) UNIQUE
);

-- Таблиця типів пального
CREATE TABLE main.dim_fuel_types (
    id SERIAL PRIMARY KEY,
    fuel_type VARCHAR(50) UNIQUE
);

-- Таблиця трансмісій
CREATE TABLE main.dim_transmissions (
    id SERIAL PRIMARY KEY,
    transmission VARCHAR(50) UNIQUE
);

-- Таблиця додаткових параметрів (включає пробіг)
CREATE TABLE main.dim_additional_info (
    additional_id SERIAL PRIMARY KEY,
    mileage INT
);

-- ✅ Таблиця року випуску
CREATE TABLE main.dim_year (
    year_id SERIAL PRIMARY KEY,
    production_year INT UNIQUE
);

-- Фактова таблиця цін на автомобілі з усіма зв'язками
CREATE TABLE main.fact_car_prices (
    id SERIAL PRIMARY KEY,
    car_id INT REFERENCES stage.car_price_prediction1(car_id),
    source_table VARCHAR(20), -- джерело прогнозу (prediction1, prediction2, prediction3)
    price DECIMAL(10,2),
    year_id INT REFERENCES main.dim_year(year_id), -- замість просто year
    brand_id INT REFERENCES main.dim_brands(id),
    fuel_type_id INT REFERENCES main.dim_fuel_types(id),
    transmission_id INT REFERENCES main.dim_transmissions(id),
    additional_id INT REFERENCES main.dim_additional_info(additional_id),
    model VARCHAR(50)
);

-- Зв'язок із prediction2
ALTER TABLE main.fact_car_prices
    ADD COLUMN car_prediction2_id INT REFERENCES stage.car_price_prediction2(id);

-- Зв'язок із prediction3
ALTER TABLE main.fact_car_prices
    ADD COLUMN car_prediction3_id INT REFERENCES stage.car_price_prediction3(id);


CREATE TABLE main.fact_leather_interior (
    id SERIAL PRIMARY KEY,
    car_price_id INT REFERENCES main.fact_car_prices(id),
    leather_interior BOOLEAN
);

-- Фактова таблиця по локації
CREATE TABLE main.fact_locations (
    id SERIAL PRIMARY KEY,
    car_price_id INT REFERENCES main.fact_car_prices(id),
    location VARCHAR(50)
);


SELECT table_schema, table_name
FROM information_schema.tables
WHERE table_schema IN ('stage', 'main');