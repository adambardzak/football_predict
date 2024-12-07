import tensorflow as tf
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objs as go

# Quick test
test_data = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})


# Create a simple plot to test plotly
fig = go.Figure(data=go.Scatter(x=test_data['A'], y=test_data['B']))
fig.show()