# Galaxy Cluster Redshift Classification Using Machine Learning

Spring 2024 - PHYS 2550 - Final Project

*Authors: Jade Ducharme, Zacharias Escalante, FeiFan Deng, Soren Helhoski, Shi Yan*

## Structure

#### Preprocessing

First, the data is loaded, basic preprocessing steps are applied (taking care of NaN values, normalization, etc. --> TODO), and the data is split into training, validation, and test sets. We visualize our input data using histograms and 2D feature vs. label (redshift) plots.

#### Simple FCNN

As a first step, we want to know how a simple fully-connected neural network performs on this dataset. We define a simple FCNN model in ```model.py```, train it, and visualize the results using a predicted label (redshift) vs. true label (redshift) plot.

#### 1D CNN

TODO.

#### GAT

We implement a Graph Attention Network with the attentional mechanism described in Veličković et al. (2018).

#### KNN Regression 

We perform a k-nearest neighbors regression on the cleaned data, and compare the results to spectroscopic data for varying k values and feature normalizations.
