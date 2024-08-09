import streamlit as st 
from util import Social

# Function to manage navigation
def navigate(page):
    st.query_params["page"]=page


def SideBar():
    # Sidebar layout with navigation
    with st.sidebar:
        
        st.image("booms.png", use_column_width=True)
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
        
        if st.button("🛡️ Home"):
            navigate("Home")
        
        if st.button("🛡️ Yolo for Logo"):
            navigate("yolologo")
            
            
        if st.button("🐙 GitHub Stats"):
            navigate("github_stats")
            
        if st.button("🚀 Social"):
            navigate("Social")
            
        if st.button("🤖 Straddle Bot"):
            navigate("straddle_bot")
        if st.button("⚙️ Automate"):
            navigate("automate")
        if st.button("🛒 Order System"):
            navigate("order_system")
        if st.button("🌐 Multi Account"):
            navigate("multi_account")
        if st.button("🕵️‍♂️ PriceAction Backtester"):
            navigate("priceaction_backtester")
        if st.button("👼 Oauth Angel"):
            navigate("oauth_angel")
        if st.button("🏦 Oauth IIFL"):
            navigate("oauth_iifl")
        if st.button("🎃 Oauth Upstox"):
            navigate("oauth_upstox")

        