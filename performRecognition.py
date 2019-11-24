# Import the modules
import cv2
# from sklearn.externals import joblib
# from skimage.feature import hog
import numpy as np
from keras.models import load_model
# import modelTrain` as mTT
# import h5py
# from modelTrain import test_images, test_labels


# Load the classifier
# clf = joblib.load("digits_cls.pkl")
try:
    test = load_model('MNIST.h5')
except:
    print('Error')
    exit(0)

