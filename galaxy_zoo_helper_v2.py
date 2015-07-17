

from __future__ import division

import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import mahotas, cv2, math
from scipy import ndimage
from skimage.filter import threshold_otsu
from EllipseFitter import EllipseFitter
from sklearn.cluster import KMeans
from scipy.stats import describe
from sklearn.base import clone

def replace_with_feature_means(ar):
	## use this to process the angles_and_sides features to replace
	## missing or na values represented as 1234554321 with mean feature values
	arc = np.copy(ar)
	for i in range(arc.shape[1]):
		replace_value = np.mean(arc[arc[:,i] != 1234554321,i])
		arc[arc[:,i] == 1234554321,i] = replace_value
	return arc


def get_ellipse_axes_p(images_dir, id_number, s = 140):
	## pass in id of the image
	## value of s = 120 seems to be the best so far based on trial and error
	## s = 140 seems to be better than 120 but i have apprehensions that it might be too small an image
	im = mahotas.imread(images_dir + str(id_number) + ".jpg", as_grey = True)
	im = im[s:-s,s:-s]
	mask_otsu = im > threshold_otsu(im)
	label_im, nb_labels = ndimage.label(mask_otsu)
	sizes = ndimage.sum(mask_otsu, label_im, range(nb_labels + 1))
	mask_size = (sizes != sizes.max())
	remove_pixel = mask_size[label_im]
	# one_image_c = np.copy(im)
	# one_image_c[remove_pixel] = 0
	mask_otsu[remove_pixel] = 0
	pm = mahotas.labeled.bwperim(mask_otsu).astype(np.int)
	try:
		angle, major, minor, xCenter, yCenter = EllipseFitter(pm)
	except:
		## if in case there is an error in the ellipse fitting process
		## then assign these values to the major and minor axis
		## they are the means of the major and minor axis values that we have seen so far
		major, minor = 27.231031495826652, 17.67781666966156
	return major, minor, major / minor, 3.14 * major * minor

def get_ellipse_axes(images_dir, i_ds, s):
	return np.array(Parallel(n_jobs = 3)(delayed(get_ellipse_axes_p)(images_dir, i_d, s) for i_d in i_ds))

def get_haralick_of_object_p(images_dir, id_number, s = 120):
	## gaussian filter after object identification
	## this did better than the gauss filter first and then object identification
	im = mahotas.imread(images_dir + str(id_number) + ".jpg", as_grey = True)[s:-s,s:-s]
	imc = mahotas.imread(images_dir + str(id_number) + ".jpg")[s:-s,s:-s,:]
	
	mask_otsu = im > threshold_otsu(im)
	label_im, nb_labels = ndimage.label(mask_otsu)
	sizes = ndimage.sum(mask_otsu, label_im, range(nb_labels + 1))
	# return sizes
	mask_size = (sizes != sizes.max())
	remove_pixel = mask_size[label_im]
	
	imc[remove_pixel] = 0
	
	return mahotas.features.haralick(ndimage.gaussian_filter(imc,3)).mean(0)

def get_haralick_of_object(images_dir, i_ds, s):
	return Parallel(n_jobs = 3)(delayed(get_haralick_of_object_p)(images_dir, i_d, s) for i_d in i_ds)

def separate_segments_haralick_p(images_dir, id_number, s = 120, n_clust = 5):
	im = mahotas.imread(images_dir + str(id_number) + ".jpg", as_grey = True)[s:-s,s:-s]
	imc = mahotas.imread(images_dir + str(id_number) + ".jpg")[s:-s,s:-s,:]
	work = np.copy(im)
	
	mask_otsu = im > threshold_otsu(im)
	label_im, nb_labels = ndimage.label(mask_otsu)
	sizes = ndimage.sum(mask_otsu, label_im, range(nb_labels + 1))
	# return sizes
	mask_size = (sizes != sizes.max())
	remove_pixel = mask_size[label_im]
	
	work[remove_pixel] = 0
	# imc[remove_pixel] = 0
	
	nz_points = work.nonzero()
	
	km = KMeans(n_clust)
	km.fit(work[nz_points].reshape((work[nz_points].shape[0],1)))
	pr = km.predict(work[nz_points].reshape((work[nz_points].shape[0],1)))
	
	pr = pr + 1
	work[nz_points] = pr
	
	loop = np.arange(km.n_clusters)[(-km.cluster_centers_.flatten()).argsort()] + 1
	send = ()

	for i in loop:
		temp_loop = np.copy(imc)
		temp_loop[work != i,:] = 0
		
		ha = mahotas.features.haralick(temp_loop)
		for j in range(ha.shape[1]):
			de = describe(ha[:,j])
			send = send + de[1] + de[2:]
			
	return send

def separate_segments_haralick(images_dir, i_ds, s):
	return Parallel(n_jobs = 3)(delayed(separate_segments_haralick_p)(images_dir, i_d, s) for i_d in i_ds)

def get_data_to_do_pca_p(images_dir, id_number, s = 120, gauss = None):
	if gauss:
		im = ndimage.gaussian_filter1d(mahotas.imread(images_dir + str(id_number) + ".jpg", as_grey = True)[s:-s,s:-s], gauss)
	else:
		im = mahotas.imread(images_dir + str(id_number) + ".jpg", as_grey = True)[s:-s,s:-s]
	return im.ravel()

def get_data_to_do_pca(images_dir, i_ds, s):
	return Parallel(n_jobs = 3)(delayed(get_data_to_do_pca_p)(images_dir, i_d, s) for i_d in i_ds)

def calculate_mean_shape_size_p(images_dir, id_number, s = 120, gauss = None):
	## mostly use only 120 for this
	## 140 seems to be too exact a fit
	## add max/min, max/mean, min/mean
	im = mahotas.imread(images_dir + str(id_number) + ".jpg", as_grey = True)[s:-s,s:-s]
	if gauss:
		im = ndimage.filters.gaussian_filter1d(im, gauss)
	mask_otsu = im > threshold_otsu(im)
	label_im, nb_labels = ndimage.label(mask_otsu)
	sizes = ndimage.sum(mask_otsu, label_im, range(nb_labels + 1))
	mask_size = (sizes != sizes.max())
	remove_pixel = mask_size[label_im]

	la = np.ones(im.shape)
	la[remove_pixel] = 0
	
	pm = mahotas.labeled.bwperim(la)
	center_x, center_y = mahotas.center_of_mass(la)
	
	nz = np.transpose(np.nonzero(pm))
	
	distances = (nz[:,0] - center_x) ** 2 + (nz[:,1] - center_y) ** 2
	
	de = describe(distances)
	
	## the below one was just a check to see if x and y were interchanged. But no they aren't one.
	## distances1 = (nz[:,0] - center_y) ** 2 + (nz[:,1] - center_x) ** 2
	
	return tuple([np.sqrt(distances).mean()]) + de[1] + de[2:] + (de[1][1] / (de[1][0] +1), de[1][1] / de[2], de[1][0] / de[2])

def calculate_mean_shape_size(images_dir, i_ds, s, gauss = None):
	return np.array(Parallel(n_jobs = 3)(delayed(calculate_mean_shape_size_p)(images_dir, i_d, s, gauss) for i_d in i_ds))


def get_contour_properties(img, cnt):
	## newer version extracting more features
	## assume that only one contour is passed in
	
	area = cv2.contourArea(cnt)
	perimeter = cv2.arcLength(cnt,True)
	
	## need to include finding the centroid and the mean shape
	
	# bounding box
	bounding_box = cv2.boundingRect(cnt)
	(bx,by,bw,bh) = bounding_box
	
	bounding_box_area = bw*bh

	# aspect ratio
	aspect_ratio = bw/float(bh)

	# equivalent diameter
	equi_diameter = np.sqrt(4*area/np.pi)

	# extent = contour area/boundingrect area
	extent = area/(bw*bh)
	
	# Min Area Rectangle
	((x,y),(w,h), theta) = cv2.minAreaRect(cnt)
	
	m_rect_area = w * h
	m_rect_ratio = w / h
	
	### CONVEX HULL ###

	# convex hull
	convex_hull = cv2.convexHull(cnt)

	# convex hull area
	convex_area = cv2.contourArea(convex_hull)
	convex_perimter = cv2.arcLength(convex_hull, True)

	# solidity = contour area / convex hull area
	solidity = area/float(convex_area)
	
	### some ratio features
	
	roughness = convex_perimter / perimeter
	## circularity 
	circularity = (perimeter ** 2) / area 
	## sphericity is also called as thinness ratio
	sphericity = 4 * np.pi * area / (perimeter ** 2)
	rectangularity = area / m_rect_area
	a_2_p_ratio = area / perimeter
	
	
	## temperature
	temperature = 1 / (np.log2(2*perimeter/(perimeter-convex_perimter+1)))
	
	### ELLIPSE  ###

	ellipse = cv2.fitEllipse(cnt)

	# center, axis_length and orientation of ellipse
	(center,axes,orientation) = ellipse

	# length of MAJOR and minor axis
	majoraxis_length = max(axes)
	minoraxis_length = min(axes)
	
	## ellipse area
	e_area = np.pi * majoraxis_length * minoraxis_length
	e_ratio = majoraxis_length / minoraxis_length

	# eccentricity = sqrt( 1 - (ma/MA)^2) --- ma= minor axis --- MA= major axis
	eccentricity = np.sqrt(1-(minoraxis_length/majoraxis_length)**2)
	
	### PIXEL PARAMETERS
	
	# filledImage = np.zeros(img.shape[0:2],np.uint8)
     
	# mean value, minvalue, maxvalue
	# minval,maxval,minloc,maxloc = cv2.minMaxLoc(img,mask = filledImage)
	# meanval = cv2.mean(img,mask = filledImage)
	
	return (area, perimeter, bw, bh, bounding_box_area, aspect_ratio, equi_diameter, extent,
			m_rect_area, m_rect_ratio, convex_area, solidity, majoraxis_length, minoraxis_length,
			e_area, e_ratio, eccentricity, roughness, circularity, sphericity, rectangularity,
			a_2_p_ratio, temperature)#, minval, maxval, meanval)

def calculate_shape_properties_p(images_dir, id_number, s = 120, gauss = None):
	im = mahotas.imread(images_dir + str(id_number) + ".jpg", as_grey = True)[s:-s,s:-s]
	if gauss:
		im = ndimage.filters.gaussian_filter1d(im, gauss)
	
	mask_otsu = im > threshold_otsu(im)
	label_im, nb_labels = ndimage.label(mask_otsu)
	sizes = ndimage.sum(mask_otsu, label_im, range(nb_labels + 1))
	mask_size = (sizes != sizes.max())
	remove_pixel = mask_size[label_im]
	
	bw = np.ones(im.shape)
	bw[remove_pixel] = 0
	
	cont, hei = cv2.findContours(bw.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	if len(cont) == 1:
		cnt = cont[0]
	## try to replace this logic..
	## Basically what is happening is :
	## we are not able to identify a single object in the image
	## due to disturbances in the image 
	## example image in the training data is 448708
	else:
		contour_areas = np.array([cv2.contourArea(x) for x in cont])
		ind = (-contour_areas).argsort()[0]
		cnt = cont[ind]
	
	return get_contour_properties(im, cnt)

def calculate_shape_properties(images_dir, i_ds, s, gauss = None):
	return np.array(Parallel(n_jobs = 3)(delayed(calculate_shape_properties_p)(images_dir, i_d, s, gauss) for i_d in i_ds))

def separate_segments_properties_p(images_dir, id_number, s = 120, n_clust = 5):
	im = mahotas.imread(images_dir + str(id_number) + ".jpg", as_grey = True)[s:-s,s:-s]
	work = np.copy(im)
	
	mask_otsu = im > threshold_otsu(im)
	label_im, nb_labels = ndimage.label(mask_otsu)
	sizes = ndimage.sum(mask_otsu, label_im, range(nb_labels + 1))
	# return sizes
	mask_size = (sizes != sizes.max())
	remove_pixel = mask_size[label_im]
	
	la = np.ones(im.shape)
	la[remove_pixel] = 0
	center_x, center_y = mahotas.center_of_mass(la)
	
	work[remove_pixel] = 0
	
	nz_points = work.nonzero()
	
	km = KMeans(n_clust)
	km.fit(work[nz_points].reshape((work[nz_points].shape[0],1)))
	pr = km.predict(work[nz_points].reshape((work[nz_points].shape[0],1)))
	
	pr = pr + 1
	work[nz_points] = pr
	
	loop = np.arange(km.n_clusters)[(-km.cluster_centers_.flatten()).argsort()] + 1
	
	send = ()
	
	for i in loop:
		final = np.zeros(im.shape)
		final[work == i] = 1
		clo = mahotas.close_holes(final)
		contours, hierarchy = cv2.findContours(clo.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		# if contours:
		contour_areas = np.array([cv2.contourArea(x) for x in contours])
		ind = (-contour_areas).argsort()[0]
	
		pm = mahotas.labeled.bwperim(clo)
		nz = np.transpose(np.nonzero(pm))	
		distances = (nz[:,0] - center_x) ** 2 + (nz[:,1] - center_y) ** 2
		
		de = describe(distances)
		send = send + tuple([np.sqrt(distances).mean()]) + de[1] + de[2:] + (de[1][1] / (de[1][0] +1), de[1][1] / de[2], de[1][0] / de[2]) + get_contour_properties(im, contours[ind])

	return send

def separate_segments_properties(images_dir, i_ds, s):
	return np.array(Parallel(n_jobs = 3)(delayed(separate_segments_properties_p)(images_dir, i_d, s) for i_d in i_ds))

def get_channel_values_p(images_dir, id_number, s = 120, n_clust = 5):
	im = mahotas.imread(images_dir + str(id_number) + ".jpg", as_grey = True)[s:-s,s:-s]
	imc = mahotas.imread(images_dir + str(id_number) + ".jpg")[s:-s,s:-s]
	work = np.copy(im)
	
	mask_otsu = im > threshold_otsu(im)
	label_im, nb_labels = ndimage.label(mask_otsu)
	sizes = ndimage.sum(mask_otsu, label_im, range(nb_labels + 1))
	# return sizes
	mask_size = (sizes != sizes.max())
	remove_pixel = mask_size[label_im]
	
	work[remove_pixel] = 0
	
	nz_points = work.nonzero()
	
	km = KMeans(n_clust)
	km.fit(work[nz_points].reshape((work[nz_points].shape[0],1)))
	pr = km.predict(work[nz_points].reshape((work[nz_points].shape[0],1)))
	
	pr = pr + 1
	work[nz_points] = pr
	
	loop = np.arange(km.n_clusters)[(-km.cluster_centers_.flatten()).argsort()] + 1
	
	send = ()
	
	for i in loop:
		for j in range(3):
			de = describe(imc[work == i,j])
			send = send + de[1] + de[2:]
		
	return send

def get_channel_values(images_dir, i_ds, s):
	return np.array(Parallel(n_jobs = 3)(delayed(get_channel_values_p)(images_dir, i_d, s) for i_d in i_ds))

def calc_angle_three_points_v2(a, b, c):
	
	v_in = (a[0]-b[0], a[1]-b[1])
	v_out = (b[0]-c[0], b[1]-c[1])
	
	return math.pi - math.atan2(v_in[0]*v_out[1] - v_out[0]*v_in[1], v_in[0]*v_out[0]+v_in[1]*v_out[1])

def get_polygon_properties(appr, si):
	## send in the copy of output obtained from running a approxPolyDP function
	v = appr[:,0,:]
	v[:,1] = [si - x for x in v[:,1]]
	# print v.shape
	if v.shape[0] > 2:
		angles = [math.degrees(calc_angle_three_points_v2(x, y, z)) for x, y, z in zip(v, v[1:,:], v[2:,:])]
		angles = [math.degrees(calc_angle_three_points_v2(v[-1,:], v[0,:], v[1,:]))] + angles + [math.degrees(calc_angle_three_points_v2(v[-2,:], v[-1,:], v[0,:]))]
		
		g = np.vstack((v[1:,:], v[:1,:]))
		sides = np.sqrt(np.sum((v - g) ** 2, 1))
		
		n_vertices = v.shape[0]
		max_to_min_angle = max(angles) / (min(angles) + 1)
		max_to_sum_of_all_angles = max(angles) / (sum(angles) + 1)
		mean_abs_diff_of_angles = np.array([abs(angles[0] - angles[-1])] + [abs(x - y) for x, y in zip(angles, angles[1:])]).mean()
		max_to_min_side = sides.max() / (sides.min() + 1)
		ratio_of_sds = np.std(sides) / (np.std(angles) + 1)
		
		de1 = describe(sides)
		de2 = describe(angles)
		
		return (n_vertices,) +  de1[1] + de1[2:] + (max_to_min_side,) + de2[1] + de2[2:] + (max_to_min_angle, max_to_sum_of_all_angles, mean_abs_diff_of_angles, ratio_of_sds)
	else:
		return (1234554321,) * 18


def approx_poly_incl_segments_p(images_dir, id_number, s = 120, error = 5, n_clust = 5):
	im = mahotas.imread(images_dir + str(id_number) + ".jpg", as_grey = True)[s:-s,s:-s]
	work = np.ones(im.shape)

	mask_otsu = im > threshold_otsu(im)
	label_im, nb_labels = ndimage.label(mask_otsu)
	sizes = ndimage.sum(mask_otsu, label_im, range(nb_labels + 1))
	# return sizes
	mask_size = (sizes != sizes.max())
	remove_pixel = mask_size[label_im]

	work[remove_pixel] = 0

	contours, hierarchy = cv2.findContours(work.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	
	if len(contours) == 1:
		cnt = contours[0]
	## try to replace this logic..
	## Basically what is happening is :
	## we are not able to identify a single object in the image
	## due to disturbances in the image 
	## example image in the training data is 448708
	else:
		contour_areas = np.array([cv2.contourArea(x) for x in contours])
		ind = (-contour_areas).argsort()[0]
		cnt = contours[ind]
	
	approx = cv2.approxPolyDP(cnt,error,True)
	approx2 = np.copy(approx)
	send = get_polygon_properties(approx2, im.shape[0])

	
	work = np.copy(im)
	work[remove_pixel] = 0
	nz_points = work.nonzero()
	
	km = KMeans(n_clust)
	km.fit(work[nz_points].reshape((work[nz_points].shape[0],1)))
	pr = km.predict(work[nz_points].reshape((work[nz_points].shape[0],1)))
	
	pr = pr + 1
	work[nz_points] = pr
	
	loop = np.arange(km.n_clusters)[(-km.cluster_centers_.flatten()).argsort()] + 1

	for i in loop:
		final = np.zeros(im.shape)
		final[work == i] = 1
		clo = mahotas.close_holes(final)
		contours, hierarchy = cv2.findContours(clo.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		
		if len(contours) == 1:
			cnt = contours[0]
		## try to replace this logic..
		## Basically what is happening is :
		## we are not able to identify a single object in the image
		## due to disturbances in the image 
		## example image in the training data is 448708
		else:
			contour_areas = np.array([cv2.contourArea(x) for x in contours])
			ind = (-contour_areas).argsort()[0]
			cnt = contours[ind]
		
		approx = cv2.approxPolyDP(cnt,error,True)
		approx2 = np.copy(approx)
		send = send + get_polygon_properties(approx2, im.shape[0])
	
	return send

def approx_poly_incl_segments(images_dir, i_ds, s):
	return np.array(Parallel(n_jobs = 3)(delayed(approx_poly_incl_segments_p)(images_dir, i_d, s) for i_d in i_ds))