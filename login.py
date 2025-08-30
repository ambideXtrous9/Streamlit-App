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
        st.write("Please login to access this feature. Use Temporary Account: [username: abc, password: 123].")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Submit Login"):
            if verify_user(username, password):
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                if 'redirect_after_login' in st.session_state and st.session_state['redirect_after_login']:
                    target_page = st.session_state['redirect_after_login']
                    del st.session_state['redirect_after_login'] # Clear the redirect target
                else:
                    target_page = 'home' # Default to home page
                st.session_state['page'] = target_page
                st.query_params['page'] = target_page # Also update query params for persistence
                st.query_params["logged_in_user"] = username # Store username in URL for persistence
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid username or password")

    if st.session_state['show_signup']:
        st.subheader("Signup")
        new_username = st.text_input("New Username", key="signup_username")
        new_password = st.text_input("New Password", type="password", key="signup_password")

        if st.button("Submit Signup"):
            if new_username and new_password:  # Check if fields are not empty
                try:
                    add_user(new_username, new_password)
                    # Auto-login after successful signup
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = new_username
                    st.session_state['page'] = 'home'
                    st.query_params['page'] = 'home'
                    st.query_params["logged_in_user"] = new_username
                    st.success("Registration successful! Logging you in...")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error creating user: {str(e)}")
            else:
                st.error("Please fill in all fields")
