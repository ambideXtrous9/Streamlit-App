import streamlit as st 
from util import Social

# Function to manage navigation
def navigate(page):
    st.session_state['page'] = page
    st.query_params['page'] = page


def SideBar():
    # Sidebar layout with navigation
    
    with st.sidebar:
        
        st.image("booms.png", use_container_width=True)
        st.markdown(
            """
            <style>
            [data-testid="stImage"] img {
                border-radius: 50%;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        
        Social(sidebarPos=True)
        
        st.header("Menu")
        
        if st.button("🎃 Home"):
            navigate("Home")
            st.rerun()
            
        if st.button("💹 AI Stock Research Agent"):
            if st.session_state.get('logged_in'):
                navigate("stockscreener")
            else:
                st.session_state['redirect_after_login'] = "stockscreener"
                st.session_state['page'] = 'login'
                st.session_state['show_login'] = True
                st.session_state['show_signup'] = False
                st.rerun()

        if st.button("🔮Agent : Harry Potter X Mythology"):
            if st.session_state.get('logged_in'):
                navigate("newsqa")
            else:
                st.session_state['redirect_after_login'] = "newsqa"
                st.session_state['page'] = 'login'
                st.session_state['show_login'] = True
                st.session_state['show_signup'] = False
                st.rerun()
            
        if st.button("🚀 Yolo for Logo"):
            navigate("yolologo")
            
        if st.button("🏆 Play with Image Classifier"):
            navigate("image_classifer")
            
        if st.button("🐙 Play with Cluster"):
            navigate("clusterplay")
            
        if st.button("🌐 Social"):
            navigate("Social")

        if st.session_state.get('logged_in'):
            if st.button("Logout"):
                st.session_state['logged_in'] = False
                st.session_state['username'] = None
                st.session_state['page'] = 'home' # Reset page to home on logout
                if "logged_in_user" in st.query_params:
                    del st.query_params["logged_in_user"]
                if "page" in st.query_params:
                    del st.query_params["page"]
                st.rerun()
        else:
            login_col, signup_col = st.columns(2)
            with login_col:
                if st.button("Login"):
                    st.session_state['page'] = 'login'
                    st.session_state['show_login'] = True
                    st.session_state['show_signup'] = False
                    st.rerun()
            with signup_col:
                if st.button("Signup"):
                    st.session_state['page'] = 'signup'
                    st.session_state['show_signup'] = True
                    st.session_state['show_login'] = False
                    st.rerun()
            
        

        