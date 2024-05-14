# Galaxy Cluster Redshift Classification Using Machine Learning

Spring 2024 - PHYS 2550 - Final Project

*Authors:  Feifan Deng, Jade Ducharme, Zacharias Escalante, Soren Helhoski, Shi Yan*

The goal of our final project is to perform a linear regression task on clusters of galaxy in order to infer, based on seven distinct features, the redshift of each. 

## Structure

#### ```data``` folder

We include three files in this folder.

1. ```specz_fluxes.csv```: the "raw" data which contains some missing ("NaN") values.
2. ```clean_specz_fluxes.csv```: the "clean" data, where the missing ("NaN") values have been dealt with. This is our training data.
3. ```specz_photoz.csv```: the non_ML comparison data. Contains "true" (spectroscopic) redshifts and the associated non-ML (photoz) redshift.

#### ```src``` folder

In the ```src``` folder, you will find three accompanying Python files:

1. ```model.py```: where all model classes and training loops are defined.
2. ```preprocess.py```: where all data preprocessing functions (data loading, standardization, and splitting) are defined.
3. ```visualize.py```: where all plotting functions (loss curves, histograms, predictions vs. labels, etc.) are defined.

#### Notebooks

For a more comfortable user experience, we include two notebooks which continuously refer to the source code from the ```src``` folder, reducing code bloat in the notebooks themselves.

##### 1. ```make_clean_data.ipynb```
This notebook details how to obtain the "clean" data from the "raw" ```data/specz_fluxes.csv```.

##### 2. ```workbook.ipynb```
Here, all models are instantiated and trained, and all training and prediction curves are presented. The models we considered are:

1. Fully-Connected Neural Network
2. 1D Convolutional Neural Network
3. Graph Attention Network
4. k-Nearest-Neighbors Regression

#### Presentation Slides

We also include our final presentation slides under ```presentation_slides.pdf``` :)
