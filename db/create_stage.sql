-- Створення stage зони
CREATE SCHEMA IF NOT EXISTS stage;

-- Таблиця stage.car_price_prediction1
CREATE TABLE stage.car_price_prediction1 (
    car_id INT PRIMARY KEY,
    brand VARCHAR(50),
    year INT,
    engine_volume DECIMAL(5,2),
    fuel_type VARCHAR(50),

    gearbox_type VARCHAR(50),
    mileage INT,
    condition VARCHAR(50),
    price DECIMAL(10,2),
    model VARCHAR(50)
);

-- Таблиця stage.car_price_prediction2
CREATE TABLE stage.car_price_prediction2 (
    id INT PRIMARY KEY,
    price DECIMAL(10,2),
    levy DECIMAL(10,2),
    manufacturer VARCHAR(50),
    model VARCHAR(50),
    prod_year INT,
    category VARCHAR(50),
    leather_interior BOOLEAN,
    fuel_type VARCHAR(50),
    engine_volume DECIMAL(5,2),
    mileage INT,  -- Змінив на числовий тип
    cylinders DECIMAL(3,1),
    gearbox_type VARCHAR(50),
    drive_wheels VARCHAR(50),
    doors INT,
    wheel VARCHAR(50),
    color VARCHAR(50),
    airbags INT
);

-- Таблиця stage.car_price_prediction3
CREATE TABLE stage.car_price_prediction3 (
    id SERIAL PRIMARY KEY,
    brand VARCHAR(50),
    model VARCHAR(50),
    price DECIMAL(10,2),
    engine_volume DECIMAL(5,2),
    year INT,
    mileage INT,
    fuel VARCHAR(50),
    gearbox_type VARCHAR(50),
    location VARCHAR(50)
);