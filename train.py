
from __future__ import division

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from feature_extraction import *
from theano_mlp import train_and_get_model_props


## set the path to the folder with the competition data
## assumes that this path has the file named "training_solutions_rev1.csv" and the folder "images_training_rev1" in it
data_folder = 'G:/GREGO/Kaggle/galaxy_zoo/original_files/'
	
train_solutions = pd.read_csv(data_folder + "training_solutions_rev1.csv")
images_dir = data_folder + "images_training_rev1/"

## set the number of images to be used for training
## select a small number to check results quickly
n_rows = 10000

## extracting various features from the images

ellipse_features = get_ellipse_axes(images_dir, train_solutions['GalaxyID'][:n_rows], 140)

te = get_haralick_of_object(images_dir, train_solutions['GalaxyID'][:n_rows], 120)
haralick_features = np.vstack(te)

pca = PCA(10)

temp = get_data_to_do_pca(images_dir, train_solutions['GalaxyID'][:n_rows], 140)
pc_data = np.vstack(temp)

pc_features = pca.fit_transform(pc_data)

shape_size_features = calculate_mean_shape_size(images_dir, train_solutions['GalaxyID'][:n_rows], 120)

morph_features = calculate_shape_properties(images_dir, train_solutions['GalaxyID'][:n_rows], 120)
seg = separate_segments_properties(images_dir, train_solutions['GalaxyID'][:n_rows], 120)

colour_vals = get_channel_values(images_dir, train_solutions['GalaxyID'][:n_rows], 120)
angles_and_sides = approx_poly_incl_segments(images_dir, train_solutions['GalaxyID'][:n_rows], 120)
angles_and_sides = replace_with_feature_means(angles_and_sides)

## merging all the features into one dataset

X = np.hstack((ellipse_features, haralick_features, pc_features, shape_size_features, morph_features, seg, colour_vals, angles_and_sides))

## indexes for the best subset from the extracted features
## this was found using greedy forward algorithm while trying to optimize the evaluation metric

sub = [2, 4, 5, 6, 8, 11, 12, 13, 15, 16, 17, 20, 22, 37, 38, 41, 52, 53, 
		55, 56, 57, 58, 59, 60, 63, 69, 70, 80, 93, 96, 102, 103, 114, 116, 
		118, 119, 122, 123, 124, 125, 127, 128, 131, 135, 148, 149, 152, 156, 
		161, 164, 201, 204, 226, 232, 235, 257, 293, 296, 298, 303, 312, 349, 
		366, 378, 379, 385, 390, 398]

all_params = []
for i in range(1,38):
	all_params.append(train_and_get_model_props(X[:,sub], np.array(train_solutions.iloc[:n_rows,i]), 1e4, 5))

_ = joblib.dump(all_params, data_folder + "theano_models.pkl", compress = 9)
_ = joblib.dump(pca, data_folder + "pca_model.pkl", compress=9)
