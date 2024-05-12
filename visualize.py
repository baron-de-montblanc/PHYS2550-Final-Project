import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import networkx as nx


# ------------------ Write some input/result visualization functions -----------------------



def histogram_input(data, features, plot_feature, nbins, xrange=(0,40)):
    """
    Given input data of shape [input_size, num_features]
    and features of shape [num_features,];
    Plot the histogram for the data corresponding to the requested feature

    nbins: number of histogram bins to display (int)
    xrange: horizontal scale of histogram (tuple)
    """

    f_idx = np.where(features == plot_feature)[0][0]
    plot_data = data[:,f_idx]

    # Remove zeros since these are artifacts from when we tried to deal with nan's
    non_zero_mask = plot_data != 0
    plot_data = plot_data[non_zero_mask]

    plt.figure(figsize=(6,2))
    plt.hist(plot_data, density=True, bins=nbins, range=xrange)
    plt.title(f"Density of {plot_feature} in dataset",y=1.02)
    plt.ylabel("Density")
    plt.xlabel(plot_feature)
    plt.show()


def plot_labels_features(data, labels, features, plot_feature, yrange=None):
    """
    Given input data of shape [input_size, num_features]
    labels of shape [input_size,] and features of shape [num_features,];
    Plot the 2D distribution of label vs. plot_feature

    yrange: vertical extent of the plot. Can be either None (show the full extent) or tuple
    """

    f_idx = np.where(features == plot_feature)[0][0]
    plot_data = data[:,f_idx]

    # Remove zeros since these are artifacts from when we tried to deal with nan's
    non_zero_mask = plot_data != 0
    plot_data = plot_data[non_zero_mask]
    labels = labels[non_zero_mask]

    plt.figure(figsize=(6,2))
    plt.plot(plot_data, labels, linestyle="", marker=".", markersize=4, alpha=0.8)
    plt.title(f"Redshift as a function of {plot_feature}", y=1.02)
    plt.xlabel(plot_feature)
    plt.ylabel("Redshift")
    if yrange:
        plt.ylim(yrange[0], yrange[1])
    plt.show()



def plot_loss(loss_list, val_loss_list, transparent=False, savepath=None):
    """
    Plot the training and validation loss as a function of epoch

    transparent: (bool) If true, set facecolor to transparent and set text color to white
    savepath: (str or None) If str, save the plot to the specified path
    """

    # Set the style and text color based on transparency
    if transparent:
        text_color = 'white'
        face_color = 'none'
    else:
        text_color = 'black'
        face_color = 'white'

    # Visualize loss progression
    plt.figure(figsize=(6,2), facecolor=face_color)
    plt.plot(loss_list, label="Training loss")
    plt.plot(val_loss_list, label="Validation loss")

    plt.title("MSE Loss as a Function of Epoch", y=1.02, color=text_color)
    plt.xlabel("Epoch number", color=text_color)
    plt.ylabel("Loss", color=text_color)
    plt.tick_params(axis='both', colors=text_color)
    plt.legend()

    if savepath:
        plt.savefig(savepath, dpi=500, bbox_inches='tight')
        
    plt.show()



def visualize_predictions(true_z, predicted_z, model_name, xrange=None, yrange=None, savepath=None, transparent=False):
    """
    Given our test Subset, extract the test elements, pass them to the model and
    compare against true labels

    xrange: horizontal range of the plot. Can be either None (see full extent)
    or a tuple of ints
    yrange: vertical range of the plot. Can be either None (see full extent)
    or a tuple of ints
    savepath: if you want to save this figure, specify this save path (str)
    transparent: if True, makes the background transparent and all text white (dark mode)
    """
    # Set the style and text color based on transparency
    if transparent:
        text_color = 'white'
        face_color = 'none'
    else:
        text_color = 'black'
        face_color = 'white'

    plt.figure(figsize=(6,4), facecolor=face_color)
    plt.title(f"{model_name} Prediction vs. True Redshift", y=1.02, color=text_color)
    plt.plot(true_z, predicted_z, linestyle="", marker=".", markersize=4, label="Predictions")
    plt.xlabel("True Redshift", color=text_color)
    plt.ylabel("Predicted Redshift", color=text_color)

    if xrange:
        plt.xlim(xrange[0], xrange[1])
    if yrange:
        plt.ylim(yrange[0], yrange[1])

    # Compare to a perfect prediction slope
    xvals = np.linspace(min(true_z), max(true_z),1000)
    plt.plot(xvals, xvals, linestyle='--', color=sns.color_palette()[1], alpha=0.9, label="Perfect Predictions Line")

    # Add the model's best-fit line prediction
    slope, intercept, r, p, se = linregress(true_z, predicted_z)
    best_fit_model = slope*xvals + intercept
    plt.plot(xvals, best_fit_model, linestyle='--', color=sns.color_palette()[2], alpha=0.9, label="Best-Fit Line")

    print("----------------- Linear Regression Parameters -----------------")
    print("Slope:\t\t\t\t\t",slope)
    print("Intercept:\t\t\t\t",intercept)
    print("Coefficient of determination (R2):\t",r**2)
    print("p-value for null hypothesis:\t\t",p)
    print("Standard error on the slope:\t\t",se)

    plt.tick_params(axis='both', colors=text_color)
    plt.legend()

    if savepath:
        plt.savefig(savepath, dpi=500, bbox_inches='tight')

    plt.show()



def visualize_knn_predictions(label_list, pred_list, device, xrange=None, yrange=None):
    """
    xrange: horizontal range of the plot. Can be either None (see full extent)
    or a tuple of ints
    yrange: vertical range of the plot. Can be either None (see full extent)
    or a tuple of ints
    """

    plt.figure(figsize=(6,4))
    plt.title("Model Prediction vs. True Redshift", y=1.02)
    plt.plot(label_list, pred_list, linestyle="", marker=".", markersize=4, label="Model Predictions")
    plt.xlabel("True Redshift")
    plt.ylabel("Predicted Redshift")

    if xrange:
        plt.xlim(xrange[0], xrange[1])
    if yrange:
        plt.ylim(yrange[0], yrange[1])

    # Compare to a perfect prediction slope
    xvals = np.linspace(min(label_list), max(label_list),1000)
    plt.plot(xvals, xvals, linestyle='--', color=sns.color_palette()[1], alpha=0.9, label="Perfect Predictions Line")

    # # Add the model's best-fit line prediction
    slope, intercept, r, p, se = linregress(label_list, pred_list)

    best_fit_model = slope*xvals + intercept
    plt.plot(xvals, best_fit_model, linestyle='--', color=sns.color_palette()[2], alpha=0.9, label="Model Best-Fit Line")

    plt.legend()
    plt.show()




import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

def visualize_graph(graph, transparent=False, savepath=None):
    """
    Visualize the graphs with nodes positioned according to their feature value and index.
    
    savepath: if you want to save this figure, specify this save path (str)
    transparent: if True, makes the background transparent and all text white (dark mode)
    """
    # Set the style and text color based on transparency
    if transparent:
        text_color = 'white'
        face_color = 'none'
    else:
        text_color = 'black'
        face_color = 'white'

    # Convert to CPU for visualization if it's on GPU
    edge_index = graph.edge_index.cpu().numpy()
    features = graph.x.cpu().numpy().flatten()  # Flatten since it's 2D [7, 1]

    # Generate positions using node index for y and feature value for x
    pos = {i: [features[i], i] for i in range(len(features))}  # x is feature, y is index

    G = nx.Graph()
    for i in range(len(features)):
        G.add_node(i)
    for start, end in edge_index.T:
        G.add_edge(int(start), int(end), edge_type='knn')

    plt.figure(figsize=(6,4), facecolor=face_color)
    knn_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d['edge_type'] == 'knn']
    node_color = "#303030"
    edge_color_knn = sns.color_palette()[2]

    nx.draw_networkx_nodes(G, pos, node_size=40, node_color=node_color)
    nx.draw_networkx_edges(G, pos, edgelist=knn_edges, edge_color=edge_color_knn, width=1.5, label='kNN Edges')
    plt.title('Graph Visualization with kNN Edges', color=text_color)
    plt.xlabel("Node value", color=text_color)
    plt.ylabel("Node index", color=text_color)
    plt.legend()
    plt.tick_params(axis='both', colors=text_color)
    
    if savepath:
        plt.savefig(savepath, dpi=500, bbox_inches='tight')
    plt.show()




