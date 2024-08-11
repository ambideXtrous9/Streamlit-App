import streamlit as st 
from Clustering.clusterapp import PlotData,kmeans,DBScan,KDistGraph
from icons import glowingCluster

def showData():
    fig = PlotData()
    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)


def Cluster():
    
    showData()
    
    selected_option = st.radio("Choose Clustering Algorithm", ("K-Means", "DBSCAN"))

    if selected_option == "K-Means":
        # Slider for number of clusters
        n_clusters = st.slider("Number of clusters", 1, 10, 3)
        
        # Run button
        if st.button("Run K-Means"):
            fig = kmeans(n_clusters)
            st.plotly_chart(fig,use_container_width=True)
            
    elif selected_option == "DBSCAN":
        eps = st.slider("Epsilon (distance threshold)", 1, 40, 5)
    
        # Run button
        if st.button("Run DBSCAN"):
            fig = DBScan(eps)
            st.plotly_chart(fig)
            
            if eps <= 5 : 
                st.write('''🌟 Interesting! If All the data points are now of same color which means they are treated as noise. 
                         It is because the value of epsilon is very small and we didn’t optimize parameters. 
                         Therefore, we need to find the value of epsilon and minPoints and then train our model again.''')
                
                st.write('''💡 Would you like to see the K-Distance Graph?''')
                
                
                kfig = KDistGraph()
                st.plotly_chart(kfig)
                
                st.write('''🧠 Now chnage the eps!''')
                
            elif eps > 5 and eps < 30 :
                st.write('''💡 Change eps till 30 and see the result!''')
            
            elif eps >= 30 and eps<34:
                st.write('''🚀 DBSCAN did its job..!!''')
                
            else:
                st.write('''🛑 Don't go beyond..Outlier..!!''')
                
    glowingCluster()
