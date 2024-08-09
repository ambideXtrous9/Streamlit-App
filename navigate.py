import streamlit as st 
from util import GitHubStats, HomePage, Social, YoloforLogo


def navigator():

    # Get the current page from the query parameters
    
    page = "page"
    
    if page in st.query_params.keys():
        page = st.query_params["page"]
    else: page = "home"

    # Display content based on the selected menu item
    if page == "home":
        HomePage()
        

    elif page == "Home":
        HomePage()

    # Page content based on the selected menu item
    elif page == "yolologo":
        st.title(":rainbow[YOLO for Logo!] :sunglasses:")
        YoloforLogo()

    elif page == "github_stats":
        GitHubStats()
        

    elif page == "Social":
        Social(sidebarPos=False,heading="Social")

    elif page == "straddle_bot":
        st.subheader("Straddle Bot Page")
        st.write("Content for the Straddle Bot page goes here.")

    elif page == "automate":
        st.subheader("Automate Page")
        st.write("Content for the Automate page goes here.")

    elif page == "order_system":
        st.subheader("Order System Page")
        st.write("Content for the Order System page goes here.")

    elif page == "multi_account":
        st.subheader("Multi Account Page")
        st.write("Content for the Multi Account page goes here.")

    elif page == "priceaction_backtester":
        st.subheader("PriceAction Backtester Page")
        st.write("Content for the PriceAction Backtester page goes here.")

    elif page == "oauth_angel":
        st.subheader("Oauth Angel Page")
        st.write("Content for the Oauth Angel page goes here.")

    elif page == "oauth_iifl":
        st.subheader("Oauth IIFL Page")
        st.write("Content for the Oauth IIFL page goes here.")

    elif page == "oauth_upstox":
        st.subheader("Oauth Upstox Page")
        st.write("Content for the Oauth Upstox page goes here.")

    else:
        st.subheader("Home Page")
        st.write("Welcome to the Streamlit Portfolio. Use the sidebar to navigate.")
