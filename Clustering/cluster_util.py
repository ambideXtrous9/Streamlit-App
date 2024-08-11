import streamlit as st 
from Clustering.clusterapp import PlotData

def showData():
    fig = PlotData()
    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=False)
