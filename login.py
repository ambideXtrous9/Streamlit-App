import streamlit as st
from auth_utils import add_user, verify_user, create_users_table

def login_page():
    create_users_table()
    st.title("Login / Signup")

    choice = st.selectbox("Login or Signup", ["Login", "Signup"])

    if choice == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if verify_user(username, password):
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid username or password")

    else:
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")

        if st.button("Signup"):
            add_user(new_username, new_password)
