KaggleGalaxyZoo
===============

My submission to the Kaggle competition "Galaxy Zoo". The goal of the competition was to predict 
the probability distributions of geometric properties of galaxies from their images

My best submission produced a score of ~ `0.107` and landed me at 65th position at the end of the competition, under the team name `Milchstraße`.
Was ok with my performance given that this was my first time working with images and also that I did not know about ConvNets back then

The data for the competition can be downloaded from `https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data`

The `train.py` code will have to be run first. This will extract various geometric, segment related and colour based features
from the images using computer vision techniques. The best subset from these features are used to train a single layer neural network, one for
each output class to be predicted. These models are pickled to the data folder.

Next `predict_test_images.py` is run which extracts the necessary features from the test images and re-loads the pickled models and makes the 
final predictions using these.


Dependencies
===============

numpy  
scipy  
pandas  
sklearn  

