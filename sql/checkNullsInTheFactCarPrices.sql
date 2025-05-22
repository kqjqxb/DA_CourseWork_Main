-- Перевірка записів, де всі ідентифікатори NULL
SELECT COUNT(*) AS null_all_ids
FROM main.fact_car_prices
WHERE car_id IS NULL AND car_prediction2_id IS NULL AND car_prediction3_id IS NULL;

-- Видалення проблемних записів (якщо такі є)
DELETE FROM main.fact_car_prices
WHERE car_id IS NULL AND car_prediction2_id IS NULL AND car_prediction3_id IS NULL;

-- Повторна перевірка розподілу за source_table
SELECT
    source_table,
    COUNT(*) AS record_count,
    SUM(CASE WHEN car_id IS NULL THEN 1 ELSE 0 END) AS null_car_id,
    SUM(CASE WHEN car_prediction2_id IS NULL THEN 1 ELSE 0 END) AS null_car_prediction2_id,
    SUM(CASE WHEN car_prediction3_id IS NULL THEN 1 ELSE 0 END) AS null_car_prediction3_id
FROM main.fact_car_prices
GROUP BY source_table;


SELECT
    COUNT(*) AS null_all_ids
FROM main.fact_car_prices
WHERE car_id IS NULL AND car_prediction2_id IS NULL AND car_prediction3_id IS NULL;