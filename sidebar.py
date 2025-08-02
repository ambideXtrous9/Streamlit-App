import streamlit as st 
from util import Social

# Function to manage navigation
def navigate(page):
    st.query_params["page"]=page


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
            
        if st.button("💹 AI Stock Research Agent"):
            navigate("stockscreener")

        if st.button("🔮Agent : Harry Potter X Mythology"):
            navigate("newsqa")
            
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
                st.rerun()
        else:
            if st.button("Login"):
                navigate("login")
            
        

        