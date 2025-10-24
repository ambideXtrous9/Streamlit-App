import streamlit as st 
from Clustering.clusterapp import PlotData,kmeans,DBScan,KDistGraph
from icons import glowingCluster


def showData():
    fig = PlotData()
    # Display in Streamlit
    st.plotly_chart(fig, width='stretch',config={'displayModeBar': False})



def Cluster():
    # Display the data
    showData()
    
    # Store selected option in session state
    if 'selected_option' not in st.session_state:
        st.session_state.selected_option = "K-Means"
        
    # Store clustering parameters in session state
    if 'n_clusters' not in st.session_state:
        st.session_state.n_clusters = 3
    if 'eps' not in st.session_state:
        st.session_state.eps = 5
    
    # Option to choose clustering algorithm
    selected_option = st.radio("Choose Clustering Algorithm", ("K-Means", "DBSCAN"), key='algorithm_radio')
    st.session_state.selected_option = selected_option
    
    if selected_option == "K-Means":
        # Slider for number of clusters
        n_clusters = st.slider("Number of clusters", 1, 10, st.session_state.n_clusters)
        
        if n_clusters != st.session_state.get('prev_n_clusters', None):
            st.session_state.n_clusters = n_clusters
            st.session_state.prev_n_clusters = n_clusters
            st.rerun()
            
        # Run K-Means with current parameters
        fig = kmeans(st.session_state.n_clusters)
        st.plotly_chart(fig, width='stretch',config={'displayModeBar': False})
        st.write('''💡 Why don't you try DBSCAN..!!''')
        
    elif selected_option == "DBSCAN":
        # Slider for epsilon (distance threshold)
        eps = st.slider("Epsilon (distance threshold)", 1, 40, st.session_state.eps)
        
        if eps != st.session_state.get('prev_eps', None):
            st.session_state.eps = eps
            st.session_state.prev_eps = eps
            st.rerun()
            
        # Run DBSCAN with current parameters
        fig = DBScan(st.session_state.eps)
        st.plotly_chart(fig, width='stretch',config={'displayModeBar': False})
        
        # Display additional information based on epsilon value
        if eps <= 5: 
            st.write('''🌟 Interesting! If all the data points are now of the same color, it means they are treated as noise. 
                         It is because the value of epsilon is very small and we didn’t optimize parameters. 
                         Therefore, we need to find the value of epsilon and minPoints and then train our model again.''')
            
            # Option to see the K-Distance Graph
            show_k_graph = st.radio('💡 Would you like to see the K-Distance Graph?', ('Yes', 'No'), index=1)
            if show_k_graph == 'Yes':
                kfig = KDistGraph()
                st.plotly_chart(kfig, width='stretch',config={'displayModeBar': False})
                st.write('''🧠 Now change the eps and see the result!''')
        
        elif eps > 5 and eps < 30:
            # Option to see the K-Distance Graph
            show_k_graph = st.radio('💡 Would you like to see the K-Distance Graph?', ('Yes', 'No'), index=1)
            if show_k_graph == 'Yes':
                kfig = KDistGraph()
                st.plotly_chart(kfig, width='stretch',config={'displayModeBar': False})

            st.write('''💡 Change eps till 30 and see the result!''')
        
        elif eps >= 30 and eps < 34:
            st.write('''🚀 DBSCAN did its job..!!''')
        
        else:
            st.write('''🛑 Don't go beyond.. Outlier..!!''')
    
    glowingCluster()
