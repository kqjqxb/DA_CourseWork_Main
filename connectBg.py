import psycopg2
conn = psycopg2.connect(database="data_analysis", user="postgres", password="09864542", host="localhost", port="5433")
print("Database Connected....")