
from __future__ import division

import os
import numpy as np
import pandas as pd
from sklearn.externals import joblib
import theano.tensor as T
import theano
from theano.tensor import nnet as NN
from feature_extraction import *

def process_test(names, indexes, trained_models_mlp, pc_model, output_path, subset):
	# feed in array of image ids
	sub_names = names[indexes:indexes+1000]
	
	ellipse_features = get_ellipse_axes(test_images_dir, sub_names, 140)
	te = get_haralick_of_object(test_images_dir, sub_names, 120)
	haralick_features = np.vstack(te)
	
	temp = get_data_to_do_pca(test_images_dir, sub_names, 140)
	pc_data = np.vstack(temp)
	pc_features = pc_model.transform(pc_data)
	
	shape_size_features = calculate_mean_shape_size(test_images_dir, sub_names, 120)
	morph_features = calculate_shape_properties(test_images_dir, sub_names, 120)
	seg = separate_segments_properties(test_images_dir, sub_names, 120)
	colour_vals = get_channel_values(test_images_dir, sub_names, 120)
	angles_and_sides = approx_poly_incl_segments(test_images_dir, sub_names, 120)
	angles_and_sides = replace_with_feature_means(angles_and_sides)
	
	X = np.hstack((ellipse_features, haralick_features, pc_features, shape_size_features, morph_features, seg, colour_vals, angles_and_sides))
	
	output = pd.DataFrame({"GalaxyID": sub_names})
	columns = [u'Class1.1', u'Class1.2', u'Class1.3', u'Class2.1', u'Class2.2', u'Class3.1', u'Class3.2', u'Class4.1', u'Class4.2', u'Class5.1', u'Class5.2', u'Class5.3', u'Class5.4', u'Class6.1', u'Class6.2', u'Class7.1', u'Class7.2', u'Class7.3', u'Class8.1', u'Class8.2', u'Class8.3', u'Class8.4', u'Class8.5', u'Class8.6', u'Class8.7', u'Class9.1', u'Class9.2', u'Class9.3', u'Class10.1', u'Class10.2', u'Class10.3', u'Class11.1', u'Class11.2', u'Class11.3', u'Class11.4', u'Class11.5', u'Class11.6']

	x1 = T.matrix('x1')
	W1 = T.matrix('W1')
	b1 = T.dvector('b1')
	W2 = T.matrix('W2')
	b2 = T.dvector('b2')
	fnt1 = T.tanh(T.dot(x1, W1) + b1)
	fnt = theano.function(inputs = [x1, W1, b1, W2, b2], outputs = T.flatten(NN.sigmoid(T.dot(fnt1, W2) + b2)))
	for i in range(len(trained_models_mlp)):
		preds = fnt(trained_models_mlp[i][4].transform(X[:,subset]), trained_models_mlp[i][0], trained_models_mlp[i][1], trained_models_mlp[i][2], trained_models_mlp[i][3])
		output[columns[i]] = preds
	
	output.to_csv(output_path + str(indexes) + ".csv", index = False, encoding = 'utf-8')

def join_together(source_folder, ouput_file_name):
	file_names = os.listdir(source_folder)
	final = pd.DataFrame()
	for file_name in file_names:
		final = final.append(pd.read_csv(source_folder + file_name, encoding = 'utf-8'))
	final.to_csv(ouput_file_name, index = False, encoding = 'utf-8', cols = [u"GalaxyID", u'Class1.1', u'Class1.2', u'Class1.3', u'Class2.1', u'Class2.2', u'Class3.1', u'Class3.2', u'Class4.1', u'Class4.2', u'Class5.1', u'Class5.2', u'Class5.3', u'Class5.4', u'Class6.1', u'Class6.2', u'Class7.1', u'Class7.2', u'Class7.3', u'Class8.1', u'Class8.2', u'Class8.3', u'Class8.4', u'Class8.5', u'Class8.6', u'Class8.7', u'Class9.1', u'Class9.2', u'Class9.3', u'Class10.1', u'Class10.2', u'Class10.3', u'Class11.1', u'Class11.2', u'Class11.3', u'Class11.4', u'Class11.5', u'Class11.6'])


data_folder = '/path/to/competition/data/'
test_images_dir = data_folder + 'images_test_rev1/'

names = sorted([int(x[:6]) for x in os.listdir(test_images_dir)])


sub = [2, 4, 5, 6, 8, 11, 12, 13, 15, 16, 17, 20, 22, 37, 38, 41, 52, 53, 
		55, 56, 57, 58, 59, 60, 63, 69, 70, 80, 93, 96, 102, 103, 114, 116, 
		118, 119, 122, 123, 124, 125, 127, 128, 131, 135, 148, 149, 152, 156, 
		161, 164, 201, 204, 226, 232, 235, 257, 293, 296, 298, 303, 312, 349, 
		366, 378, 379, 385, 390, 398]

all_params_mlp = joblib.load(data_folder + "theano_models.pkl")
pca = joblib.load(data_folder + "pca_model.pkl")

if not os.path.exists(data_folder + "submission_part_files/"):
	os.makedirs(data_folder + "submission_part_files/")

for i in range(0, 80000, 1000):
	process_test(names, i, all_params_mlp, pca, data_folder + "submission_part_files/part_", sub)
	print i

join_together(data_folder + "submission_part_files/", data_folder + "submission.csv")
