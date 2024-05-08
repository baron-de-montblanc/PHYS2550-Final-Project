import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress


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

    plt.figure(figsize=(6,4))
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

    plt.figure(figsize=(6,4))
    plt.plot(plot_data, labels, linestyle="", marker=".", markersize=4, alpha=0.8)
    plt.title(f"Redshift as a function of {plot_feature}", y=1.02)
    plt.xlabel(plot_feature)
    plt.ylabel("Redshift")
    if yrange:
        plt.ylim(yrange[0], yrange[1])
    plt.show()



def plot_loss(loss_list, val_loss_list):

    # Visualize loss progression
    plt.figure(figsize=(6,4))
    plt.plot(loss_list, label="Training loss")
    plt.plot(val_loss_list, label="Validation loss")

    plt.title("MSE Loss as a Function of Epoch", y=1.02)
    plt.xlabel("Epoch number")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_accuracy(acc_list, val_acc_list):
    acc_list = [i.item() for i in acc_list]
    val_acc_list = [i.item() for i in val_acc_list]
    plt.figure(figsize=(6,4))
    plt.plot(acc_list, label="Training accuracy")
    plt.plot(val_acc_list, label="Validation accuracy")

    plt.title("R2-Score as a Function of Epoch", y=1.02)
    plt.xlabel("Epoch number")
    plt.ylabel("R2-Score")
    plt.legend()
    plt.show()



def visualize_predictions(test_set, model, device, xrange=None, yrange=None):
    """
    Given our test Subset, extract the test elements, pass them to the model and
    compare against true labels

    xrange: horizontal range of the plot. Can be either None (see full extent)
    or a tuple of ints
    yrange: vertical range of the plot. Can be either None (see full extent)
    or a tuple of ints
    """
    label_list = []
    pred_list = []
    for data, label in test_set:
        data = data.to(device)
        label_list.append(label.item())
        pred = model(data)
        pred_list.append(pred.item())

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

    # Add the model's best-fit line prediction
    slope, intercept, r, p, se = linregress(label_list, pred_list)

    best_fit_model = slope*xvals + intercept
    plt.plot(xvals, best_fit_model, linestyle='--', color=sns.color_palette()[2], alpha=0.9, label="Model Best-Fit Line")

    plt.legend()
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

