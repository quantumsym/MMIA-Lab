#!/usr/bin/env python

import numpy as np
import plotly.graph_objects as go
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import plotly.io as pio

# Set the theme
pio.templates.default = "perplexity"

# Generate synthetic 2D binary classification data
np.random.seed(42)
X, y = make_classification(n_samples=300, n_features=2, n_redundant=0, 
                          n_informative=2, n_clusters_per_class=1, 
                          random_state=42, class_sep=0.8)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train MLP
mlp = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

# Calculate accuracies
train_accuracy = mlp.score(X_train, y_train)
test_accuracy = mlp.score(X_test, y_test)

# Create decision boundary
h = 0.05  # step size
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict on mesh
mesh_points = np.c_[xx.ravel(), yy.ravel()]
Z = mlp.predict_proba(mesh_points)[:, 1]
Z = Z.reshape(xx.shape)

# Create the plot
fig = go.Figure()

# Add decision boundary contour
fig.add_trace(go.Contour(
    x=np.arange(x_min, x_max, h),
    y=np.arange(y_min, y_max, h),
    z=Z,
    colorscale=[[0, '#1FB8CD'], [0.5, 'white'], [1, '#DB4545']],
    opacity=0.3,
    showscale=False,
    contours=dict(
        start=0,
        end=1,
        size=0.1,
        showlines=False
    ),
    hovertemplate='Prob: %{z:.2f}<extra></extra>'
))

# Add decision boundary line at 0.5 probability
fig.add_trace(go.Contour(
    x=np.arange(x_min, x_max, h),
    y=np.arange(y_min, y_max, h),
    z=Z,
    contours=dict(
        start=0.5,
        end=0.5,
        size=0.1,
        coloring='lines'
    ),
    line=dict(color='black', width=3),
    showscale=False,
    hoverinfo='skip',
    name='Decision Bound'
))

# Add training data points
class_0_train = X_train[y_train == 0]
class_1_train = X_train[y_train == 1]

fig.add_trace(go.Scatter(
    x=class_0_train[:, 0],
    y=class_0_train[:, 1],
    mode='markers',
    marker=dict(color='#1FB8CD', size=8, symbol='circle'),
    name='Class 0 (Train)',
    hovertemplate='X1: %{x:.2f}<br>X2: %{y:.2f}<br>Class: 0<extra></extra>'
))

fig.add_trace(go.Scatter(
    x=class_1_train[:, 0],
    y=class_1_train[:, 1],
    mode='markers',
    marker=dict(color='#DB4545', size=8, symbol='circle'),
    name='Class 1 (Train)',
    hovertemplate='X1: %{x:.2f}<br>X2: %{y:.2f}<br>Class: 1<extra></extra>'
))

# Add test data points (with different symbols)
class_0_test = X_test[y_test == 0]
class_1_test = X_test[y_test == 1]

fig.add_trace(go.Scatter(
    x=class_0_test[:, 0],
    y=class_0_test[:, 1],
    mode='markers',
    marker=dict(color='#1FB8CD', size=8, symbol='diamond'),
    name='Class 0 (Test)',
    hovertemplate='X1: %{x:.2f}<br>X2: %{y:.2f}<br>Class: 0<extra></extra>'
))

fig.add_trace(go.Scatter(
    x=class_1_test[:, 0],
    y=class_1_test[:, 1],
    mode='markers',
    marker=dict(color='#DB4545', size=8, symbol='diamond'),
    name='Class 1 (Test)',
    hovertemplate='X1: %{x:.2f}<br>X2: %{y:.2f}<br>Class: 1<extra></extra>'
))

# Update layout
fig.update_layout(
    title=f'MLP Binary Class (Test Acc: {test_accuracy:.2f})',
    xaxis_title='Feature 1',
    yaxis_title='Feature 2',
    legend=dict(
        orientation='v',
        yanchor='top',
        y=1,
        xanchor='left',
        x=1.02
    )
)

fig.update_xaxes(range=[x_min, x_max])
fig.update_yaxes(range=[y_min, y_max])

# Only apply cliponaxis to scatter traces
for i, trace in enumerate(fig.data):
    if trace.type == 'scatter':
        trace.update(cliponaxis=False)

# Save the chart
fig.write_image("mlp_classification.png")
