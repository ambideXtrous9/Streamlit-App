import streamlit as st 
from util import Social

stock = "https://cdn-icons-gif.flaticon.com/17507/17507028.gif"
harry = "https://64.media.tumblr.com/e5e401e35d609e217c19a24204360b8d/tumblr_mg3h0yvGFD1rgpyeqo1_500.gif"
yolo = "https://images.squarespace-cdn.com/content/v1/5a42a3000abd044bd3244bf2/1551247107452-HYAEHY39IKJ2LJTGNLQR/YOLO-Lettering-Sticker-Joan-Quiros.gif"
imgclassifier = "https://mlnotebook.github.io/img/CNN/poolfig.gif"
cluster = "https://cdn.dribbble.com/userupload/20456242/file/original-f31f3824dec1d33b1abf5895ce03de45.gif"
boom = "boom.png"
mcpairbnb = "mcp_airbnb.png"

# Function to manage navigation
def navigate(page):
    st.session_state['page'] = page
    st.query_params['page'] = page


def SideBar():
    # Initialize session_state for sidebar image
    if "sidebar_image" not in st.session_state:
        st.session_state["sidebar_image"] = boom  # Default for Home

    with st.sidebar:
        # Show dynamic image
        st.image(st.session_state["sidebar_image"], width='stretch')

        st.markdown(
            """
            <style>
            [data-testid="stImage"] img {   
                object-fit: cover;   
                border-radius: 50%;  
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        Social(sidebarPos=True)

        st.header("Menu")

        # 🎃 Home
        if st.button("🎃 Home"):
            st.session_state["sidebar_image"] = boom
            navigate("Home")
            st.rerun()

        # 💹 AI Stock Research Agent
        if st.button("💹 AI Stock Research Agent"):
            st.session_state["sidebar_image"] = stock  # your custom image
            if st.session_state.get('logged_in'):
                navigate("stockscreener")
            else:
                st.session_state['redirect_after_login'] = "stockscreener"
                st.session_state['page'] = 'login'
                st.session_state['show_login'] = True
                st.session_state['show_signup'] = False
                st.rerun()

        # 🔮 Agent : Harry Potter X Mythology
        if st.button("🪄 Harry Potter X Mythology"):
            st.session_state["sidebar_image"] = harry
            if st.session_state.get('logged_in'):
                navigate("newsqa")
                st.rerun()
            else:
                st.session_state['redirect_after_login'] = "newsqa"
                st.session_state['page'] = 'login'
                st.session_state['show_login'] = True
                st.session_state['show_signup'] = False
                st.rerun()

        # 🔮 Agent : MCP Powered Airbnb Tour Agent
        if st.button("🏡 Tour Agent"):
            st.session_state["sidebar_image"] = mcpairbnb
            if st.session_state.get('logged_in'):
                navigate("tourAgent")
                st.rerun()
            else:
                st.session_state['redirect_after_login'] = "tourAgent"
                st.session_state['page'] = 'login'
                st.session_state['show_login'] = True
                st.session_state['show_signup'] = False
                st.rerun()

        # 🚀 Yolo for Logo
        if st.button("🚀 Yolo for Logo"):
            st.session_state["sidebar_image"] = yolo
            navigate("yolologo")
            st.rerun()

        # 🏆 Play with Image Classifier
        if st.button("🏆 Play with Image Classifier"):
            st.session_state["sidebar_image"] = imgclassifier
            navigate("image_classifer")
            st.rerun()

        # 🐙 Play with Cluster
        if st.button("🐙 Play with Cluster"):
            st.session_state["sidebar_image"] = cluster
            navigate("clusterplay")
            st.rerun()

        # 🌐 Social
        if st.button("🌐 Social"):
            st.session_state["sidebar_image"] = boom
            navigate("Social")
            st.rerun()

        # Login / Logout handling
        if st.session_state.get('logged_in'):
            if st.button("Logout"):
                st.session_state['logged_in'] = False
                st.session_state['username'] = None
                st.session_state['page'] = 'home'
                st.session_state["sidebar_image"] = boom  # back to default
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
                    st.session_state["sidebar_image"] = boom
                    st.rerun()
            with signup_col:
                if st.button("Signup"):
                    st.session_state['page'] = 'signup'
                    st.session_state['show_signup'] = True
                    st.session_state['show_login'] = False
                    st.session_state["sidebar_image"] = boom
                    st.rerun()
