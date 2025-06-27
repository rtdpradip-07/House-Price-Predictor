import sqlite3
import pandas as pd

DB_PATH = "user_data.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
                    username TEXT,
                    area REAL,
                    bedrooms INTEGER,
                    bathrooms INTEGER,
                    stories INTEGER,
                    parking INTEGER,
                    price REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )''')
    conn.commit()
    conn.close()

def save_prediction(username, area, bedrooms, bathrooms, stories, parking, price):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO predictions (username, area, bedrooms, bathrooms, stories, parking, price) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (username, area, bedrooms, bathrooms, stories, parking, price))
    conn.commit()
    conn.close()

def get_user_history(username):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM predictions WHERE username = ?", conn, params=(username,))
    conn.close()
    return df
