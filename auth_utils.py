import sqlite3
import streamlit as st
import bcrypt

def get_db_connection():
    conn = sqlite3.connect('users.db')
    return conn

def create_users_table():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
    """)
    # Add a dummy admin user if it doesn't exist
    c.execute("SELECT * FROM users WHERE username = 'admin'")
    if not c.fetchone():
        hashed_password = bcrypt.hashpw('admin'.encode('utf-8'), bcrypt.gensalt())
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", ('admin', hashed_password.decode('utf-8')))
    conn.commit()
    conn.close()

def add_user(username, password):
    conn = get_db_connection()
    c = conn.cursor()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password.decode('utf-8')))
        conn.commit()
        st.success("You have successfully signed up!")
    except sqlite3.IntegrityError:
        st.error("Username already exists.")
    finally:
        conn.close()

def verify_user(username, password):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    if result:
        hashed_password = result[0].encode('utf-8')
        if bcrypt.checkpw(password.encode('utf-8'), hashed_password):
            return True
    return False
