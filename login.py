import streamlit as st
from auth_utils import add_user, verify_user, create_users_table

def login_page():
    create_users_table()

    if 'show_login' not in st.session_state:
        st.session_state['show_login'] = False
    if 'show_signup' not in st.session_state:
        st.session_state['show_signup'] = False

    if st.session_state['show_login']:
        st.subheader("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Submit Login"):
            if verify_user(username, password):
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.session_state['page'] = 'home' # Redirect to home page after successful login
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid username or password")

    if st.session_state['show_signup']:
        st.subheader("Signup")
        new_username = st.text_input("New Username", key="signup_username")
        new_password = st.text_input("New Password", type="password", key="signup_password")

        if st.button("Submit Signup"):
            add_user(new_username, new_password)
            st.success("User registered successfully! Please login.")
            st.session_state['show_signup'] = False
            st.session_state['show_login'] = True
