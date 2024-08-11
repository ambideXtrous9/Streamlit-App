import numpy as np
import math
import pandas as pd
from sklearn.neighbors import NearestNeighbors

import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans, DBSCAN
import random

np.random.seed(42)


# Function for creating datapoints in the form of a circle
def PointsInCircum(r, n=100):
    return [(math.cos(2*math.pi/n*x)*r + np.random.normal(-30, 30),
             math.sin(2*math.pi/n*x)*r + np.random.normal(-30, 30)) for x in range(1, n+1)]

def dataGen():
    # Creating data points in the form of a circle
    df = pd.DataFrame(PointsInCircum(500, 1000))
    df = df._append(PointsInCircum(300, 700))
    df = df._append(PointsInCircum(100, 300))

    # Adding noise to the dataset
    df = df._append([(np.random.randint(-600, 600), np.random.randint(-600, 600)) for i in range(300)])
    return df

    
df = dataGen()


def PlotData():
    
    # Create the plot using Plotly
    fig = go.Figure()

    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=df[0], y=df[1],
        mode='markers',
        marker=dict(
            size=8,
            color='cyan',
            opacity=0.7,
            line=dict(width=1, color='darkblue')
        )
    ))

    # Customize layout
    fig.update_layout(
        title='Dataset Visualization',
        title_font_size=24,
        xaxis_title='Feature 1',
        yaxis_title='Feature 2',
        xaxis=dict(
            range=[-650, 650],
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            range=[-650, 650],
            showgrid=False,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1
        ),
        autosize=False,
        width=800,
        height=800,
        font=dict(color='white')
    )

    # Return the figure
    return fig


# Generate a random list of colors
def generate_colors(n):
    return ["#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)]) for _ in range(n)]


def kmeans(n_clusters):
    k_means=KMeans(n_clusters=n_clusters,random_state=42)
    k_means.fit(df[[0,1]])
    df['KMeans_labels']=k_means.labels_
    # Plotting resulting clusters
    # Generate a color palette based on the number of clusters
    colors = generate_colors(n_clusters)
    
    # Create the plot using Plotly
    fig = go.Figure()

    # Add scatter plot with cluster-specific colors
    fig.add_trace(go.Scatter(
        x=df[0], y=df[1],
        mode='markers',
        marker=dict(
            size=8,
            color=[colors[label] for label in df['KMeans_labels']],  # Use the generated colors
            opacity=0.7,
            line=dict(width=1, color='darkblue')
        )
    ))

    # Customize layout
    fig.update_layout(
        title='K-Means Clustering Visualization',
        title_font_size=24,
        xaxis_title='Feature 1',
        yaxis_title='Feature 2',
        xaxis=dict(
            range=[-650, 650],
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            range=[-650, 650],
            showgrid=False,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1
        ),
        autosize=False,
        width=800,
        height=800,
        font=dict(color='white')
    )

    # Return the figure
    return fig
    
    

def DBScan(eps=5):
    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=5)
    df['DBSCAN_labels'] = dbscan.fit_predict(df[[0, 1]])

    # Generate a color palette based on the number of clusters
    unique_labels = np.unique(df['DBSCAN_labels'])
    colors = generate_colors(len(unique_labels))

    # Create the plot using Plotly
    fig = go.Figure()

    # Add scatter plot with cluster-specific colors
    fig.add_trace(go.Scatter(
        x=df[0], y=df[1],
        mode='markers',
        marker=dict(
            size=8,
            color=[colors[label] if label != -1 else 'grey' for label in df['DBSCAN_labels']],  # Use the generated colors
            opacity=0.7,
            line=dict(width=1, color='darkblue')
        )
    ))

    # Customize layout
    fig.update_layout(
        title='DBSCAN Clustering Visualization',
        title_font_size=24,
        xaxis_title='Feature 1',
        yaxis_title='Feature 2',
        xaxis=dict(
            range=[-650, 650],
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            range=[-650, 650],
            showgrid=False,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1
        ),
        autosize=False,
        width=800,
        height=800,
        font=dict(color='white')
    )

    # Return the figure
    return fig
    


# Function to create K-Dist Graph using Plotly
def KDistGraph():
    # Calculate the k-distances
    neigh = NearestNeighbors(n_neighbors=5)
    nbrs = neigh.fit(df[[0, 1]])
    distances, _ = nbrs.kneighbors(df[[0, 1]])
    
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]  # Use the second nearest neighbor's distance

    # Create the plot using Plotly
    fig = go.Figure()

    # Add line plot
    fig.add_trace(go.Scatter(
        x=np.arange(len(distances)),
        y=distances,
        mode='lines',
        line=dict(color='cyan', width=2),
        name='Epsilon'
    ))

    # Customize layout
    fig.update_layout(
        title='K-Distance Graph',
        title_font_size=24,
        xaxis_title='Data Points Sorted by Distance',
        yaxis_title='Epsilon',
        xaxis=dict(
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=False
        ),
        autosize=False,
        width=800,
        height=600,
        font=dict(color='white')
    )

    # Return the figure
    return fig