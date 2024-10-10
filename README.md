# Face-matching-using-CBIR
This project creates a CBIR model that takes query image, compares to the images in the database and results the matched images as output using image processing.
Project:
This is Face matching CBIR which takes images from database, performs feature extraction methods, and converts feature vectors, and stores in a database. Then it takes query image database, performs same operations on query images, feature vectors of query image is measured with each feature vector of database using Euclidean distance. The closest match is shown as similar matches in display based on certain threshold value of Euclidean distance.
Overview:
This project consists of a training.py script for training model and test4.py for testing the model. It contains training and testing folder containing images. You also have distance.py to plot distance metrics between inter-class and intra-class feature vectors.
While running project, first replace the file paths of training, testing and an empty directory named output.
Then, run training.py and next run test4.py.
Requirements:
Python 3.6 or over
Libraries: numpy, opencv-python, scikit-learn, scikit-image, matplotlib, scipy, pickle
