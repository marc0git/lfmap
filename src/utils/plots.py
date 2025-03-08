import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def plot_embeddings(embeddings, labels, num_classes,figure=None):
    """
    Plots 2D embeddings, colored by labels.

    Args:
    - embeddings (torch.Tensor): Tensor of 2D embeddings.
    - labels (torch.Tensor): Tensor of labels.
    - num_classes (int): Number of distinct classes in labels.
    """
    if figure is not None:
        figure.show()
        ax=figure.gca()
    # Ensure embeddings are 2D
    if embeddings.shape[1] != 2:
        raise ValueError("Embeddings must be 2D.")

    # Convert tensors to numpy arrays for plotting
    embeddings_np = embeddings.numpy()
    labels_np = labels.numpy()

    # Plotting
    if figure is None:
        plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        indices = labels_np == i
        if figure is None:
            plt.scatter(embeddings_np[indices, 0], embeddings_np[indices, 1], 5,label=f'Class {i}')
        else:
            ax.scatter(embeddings_np[indices, 0], embeddings_np[indices, 1], 5,label=f'Class {i}')
    ax.legend()
    # set title
    if figure is None:
        plt.title('Embeddings')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
    else:
        ax.set_title('Embeddings')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')

    return figure



def plot_knn_graph(data, knn_indices):
    """
    Plots the k-NN graph for 2D data.

    Args:
    data (Tensor): The input data, shape (num_points, 2).
    knn_indices (Tensor): The indices of k nearest neighbors for each point, shape (num_points, k).
    """

    plt.figure(figsize=(10, 8))
    plt.scatter(data[:, 0], data[:, 1], c='blue')

    # Draw lines to k nearest neighbors for each point
    lines = []
    for i in range(data.shape[0]):
        for j in knn_indices[i]:
            lines.append([data[i], data[j]])
    lines = LineCollection(lines, colors='red')
    # plt.plot(*zip(data[i], data[j]), c='red')
    plt.gca().add_collection(lines)

    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('k-NN Graph')
    plt.show()

def plotly_knn_graph(data, knn_indices, color='blue', figsize=(600, 600), colorscale='jet'):
    # use plotly to plot the graph  
    fig = make_subplots(rows=1, cols=1)
    x_lines = []
    y_lines = []
    for i in range(data.shape[0]):
        for j in knn_indices[i]:
            x_lines+=[data[i,0], data[j,0], None]
            y_lines+=[data[i,1], data[j,1], None]
    
    fig.add_trace(go.Scatter(x=x_lines, y=y_lines, mode='lines', name='knn', line=dict(width=0.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data[:,0], y=data[:,1], mode='markers', name='data', marker=dict(color=color, colorscale=colorscale)), row=1, col=1)

    fig.update_layout(title='k-NN Graph', xaxis_title='Dimension 1', yaxis_title='Dimension 2', width=figsize[0], height=figsize[1])
    fig.update_yaxes(  scaleanchor="x",  scaleratio=1,  )
    return fig

def plotly_knn_graphs(data1, data2, knn_indices1, knn_indices2, color1='blue', color2='blue' ,figsize=(600, 600), colorscale='jet'):
    # use plotly to plot the graph  
    fig = make_subplots(rows=1, cols=2, subplot_titles=["G1", "G2"])
    fig.update_layout(coloraxis=dict(colorscale=colorscale))

    x_lines = []
    y_lines = []
    for i in range(data1.shape[0]):
        for j in knn_indices1[i]:
            x_lines+=[data1[i,0], data1[j,0], None]
            y_lines+=[data1[i,1], data1[j,1], None]
    
    text = [str(i) for i in range(data1.shape[0])]
    # check if color is not a string
    if type(color1) != str:
        text = [f"{t}: {c}" for c,t in zip(color1, text)]
    fig.add_trace(go.Scatter(x=x_lines, y=y_lines, mode='lines', name='knn', line=dict(width=0.5 , color="grey")), row=1, col=1)
    fig.add_trace(go.Scatter(x=data1[:,0], y=data1[:,1], mode='markers', name='data', marker=dict(color=color1, coloraxis="coloraxis", size=3),
                              text= text ), row=1, col=1)

    x_lines = []
    y_lines = []
    for i in range(data2.shape[0]):
        for j in knn_indices2[i]:
            x_lines+=[data2[i,0], data2[j,0], None]
            y_lines+=[data2[i,1], data2[j,1], None]

    text = [str(i) for i in range(data1.shape[0])]
    # check if color is not a string
    if type(color1) != str:
        text = [f"{t}: {c}" for c,t in zip(color1, text)]
    fig.add_trace(go.Scatter(x=x_lines, y=y_lines, mode='lines', name='knn', line=dict(width=0.5, color="grey")), row=1, col=2)
    fig.add_trace(go.Scatter(x=data2[:,0], y=data2[:,1], mode='markers', name='data', marker=dict(color=color2, coloraxis="coloraxis", size=3),
                             text= text), row=1, col=2)

    fig.update_layout(title='k-NN Graph', xaxis_title='Dimension 1', yaxis_title='Dimension 2', width=figsize[0], height=figsize[1],
                      legend=dict(x=1, y=1.1, orientation='h', xanchor="right",))
    fig.update_yaxes(  scaleanchor="x",  scaleratio=1)
    
    return fig

