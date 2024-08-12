import numpy as np
import math
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import matplotlib.pyplot as plt
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
    # Create the plot using Seaborn and Matplotlib
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Add scatter plot
    sns.scatterplot(
        x=df[0], y=df[1],
        color='cyan',
        s=80,  # Marker size
        edgecolor='darkblue',
        linewidth=1,
        alpha=0.7,  # Marker opacity
        ax=ax  # Specify the axes to plot on
    )
    
    # Customize layout
    ax.set_title('Dataset Visualization', fontsize=24, color='white')
    ax.set_xlabel('Feature 1', fontsize=16, color='white')
    ax.set_ylabel('Feature 2', fontsize=16, color='white')
    ax.set_xlim([-650, 650])
    ax.set_ylim([-650, 650])
    ax.set_aspect('equal', adjustable='box')  # Ensure equal aspect ratio
    ax.grid(False)  # Remove gridlines
    ax.patch.set_facecolor('black')  # Set background color to black
    
    # Customize tick colors
    ax.tick_params(colors='white')
    
    # Return the figure
    return fig


def kmeans(n_clusters):
    # Apply K-Means clustering
    k_means = KMeans(n_clusters=n_clusters, random_state=42)
    k_means.fit(df[[0, 1]])
    df['KMeans_labels'] = k_means.labels_

    # Generate a color palette based on the number of clusters
    colors = sns.color_palette('hsv', n_clusters)

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.scatterplot(
        x=df[0], y=df[1],
        hue=df['KMeans_labels'],
        palette=colors,
        s=80,  # Marker size
        edgecolor='darkblue',
        linewidth=1,
        alpha=0.7,  # Marker opacity
        ax=ax  # Specify the axes to plot on
    )

    # Customize layout
    ax.set_title('K-Means Clustering Visualization', fontsize=24, color='white')
    ax.set_xlabel('Feature 1', fontsize=16, color='white')
    ax.set_ylabel('Feature 2', fontsize=16, color='white')
    ax.set_xlim([-650, 650])
    ax.set_ylim([-650, 650])
    ax.set_aspect('equal', adjustable='box')  # Ensure equal aspect ratio
    ax.grid(False)  # Remove gridlines
    ax.patch.set_facecolor('black')  # Set background color to black

    # Customize tick colors
    ax.tick_params(colors='white')

    # Remove the legend (optional)
    ax.legend([],[], frameon=False)

    # Return the figure
    return fig


def DBScan(eps=5):
    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=5)
    df['DBSCAN_labels'] = dbscan.fit_predict(df[[0, 1]])

    # Generate a color palette based on the number of clusters
    unique_labels = np.unique(df['DBSCAN_labels'])
    colors = sns.color_palette('hsv', len(unique_labels))

    # Map colors to labels
    label_color_map = {label: colors[i] if label != -1 else 'grey' for i, label in enumerate(unique_labels)}

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.scatterplot(
        x=df[0], y=df[1],
        hue=df['DBSCAN_labels'],
        palette=label_color_map,
        s=80,  # Marker size
        edgecolor='darkblue',
        linewidth=1,
        alpha=0.7,  # Marker opacity
        ax=ax  # Specify the axes to plot on
    )

    # Customize layout
    ax.set_title('DBSCAN Clustering Visualization', fontsize=24, color='white')
    ax.set_xlabel('Feature 1', fontsize=16, color='white')
    ax.set_ylabel('Feature 2', fontsize=16, color='white')
    ax.set_xlim([-650, 650])
    ax.set_ylim([-650, 650])
    ax.set_aspect('equal', adjustable='box')  # Ensure equal aspect ratio
    ax.grid(False)  # Remove gridlines
    ax.patch.set_facecolor('black')  # Set background color to black

    # Customize tick colors
    ax.tick_params(colors='white')

    # Remove the legend (optional)
    ax.legend([],[], frameon=False)

    # Return the figure
    return fig 


def KDistGraph():
    # Calculate the k-distances
    neigh = NearestNeighbors(n_neighbors=5)
    nbrs = neigh.fit(df[[0, 1]])
    distances, _ = nbrs.kneighbors(df[[0, 1]])
    
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]  # Use the second nearest neighbor's distance

    # Create the plot using Seaborn and Matplotlib
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Add line plot
    sns.lineplot(
        x=np.arange(len(distances)),
        y=distances,
        color='cyan',
        linewidth=2,
        ax=ax
    )
    
    # Customize layout
    ax.set_title('K-Distance Graph', fontsize=24, color='white')
    ax.set_xlabel('Data Points Sorted by Distance', fontsize=16, color='white')
    ax.set_ylabel('Epsilon', fontsize=16, color='white')
    
    # Customize x-axis and y-axis
    ax.set_xlim([0, len(distances)])
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5)
    
    # Set background color and remove spines
    ax.patch.set_facecolor('black')  # Set background color to black
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Customize tick colors
    ax.tick_params(colors='white')
    
    # Return the figure
    return fig