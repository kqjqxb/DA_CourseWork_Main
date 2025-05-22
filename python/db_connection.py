# db_connection.py
from sqlalchemy import create_engine

# Замініть параметри на ваші власні
sqlalchemy_engine = create_engine("postgresql+psycopg2://postgres:09864542@localhost:5433/data_analysis")