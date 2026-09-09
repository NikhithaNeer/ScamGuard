import sqlite3

DB_PATH = "C:/Users/Lenovo/OneDrive - Kent State University/Documents/ScamGuard/users.db"

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
)
""")

conn.commit()
conn.close()

print(f"users.db created successfully at: {DB_PATH}")
