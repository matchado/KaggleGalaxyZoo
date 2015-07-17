
## various miscellaneous functions that were used for analyzing the data

import matplotlib.pyplot as plt


def plot_many_images(ims):
	## gonna assume that a max of 4 plots are gonna be passed
	if len(ims) in [1, 2]:
		f, axarr = plt.subplots(1, 2)
	elif len(ims) in [3, 4]:
		f, axarr = plt.subplots(2, 2)
	axarr = axarr.ravel()
	
	for i in range(len(ims)):
		if len(ims[i].shape) == 2:
			axarr[i].imshow(ims[i], cmap = 'gray')
		else:
			axarr[i].imshow(ims[i])
			
	plt.show()

