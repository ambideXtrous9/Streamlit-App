import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

import plotly.express as px

# Function for creating datapoints in the form of a circle
def PointsInCircum(r, n=100):
    return [(math.cos(2*math.pi/n*x)*r + np.random.normal(-30, 30),
             math.sin(2*math.pi/n*x)*r + np.random.normal(-30, 30)) for x in range(1, n+1)]

def PlotData():
    # Creating data points in the form of a circle
    df = pd.DataFrame(PointsInCircum(500, 1000))
    df = df._append(PointsInCircum(300, 700))
    df = df._append(PointsInCircum(100, 300))

    # Adding noise to the dataset
    df = df._append([(np.random.randint(-600, 600), np.random.randint(-600, 600)) for i in range(300)])

    # Plotting using Plotly
    fig = px.scatter(df, x=0, y=1, title='Dataset', labels={'0': 'Feature 1', '1': 'Feature 2'})
    return fig

