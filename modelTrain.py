from keras.datasets import mnist
from keras.utils import to_categorical
from keras import layers
from keras import models
import numpy as np
import cv2
import h5py

modal = models.Sequential()
modal.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
modal.add(layers.MaxPooling2D((2, 2)))
modal.add(layers.Conv2D(64, (3, 3), activation='relu'))
modal.add(layers.MaxPooling2D((2, 2)))
modal.add(layers.Conv2D(64, (3, 3), activation='relu'))
modal.add(layers.Flatten())
modal.add(layers.Dense(64, activation='relu'))
modal.add(layers.Dense(10, activation='softmax'))


# im = cv2.imread("maze00.jpg")
# im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# rects = [(44, 44, 32, 32), (164, 44, 32, 32), (164, 124, 32, 32),
#          (284, 124, 32, 32), (164, 204, 32, 32), (84, 324, 32, 32)]

# for rect in rects:
#     leng = 32
#     pt1 = int(rect[1])
#     pt2 = int(rect[0])
#     # roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
#     roi = im_gray[pt1:pt1+leng, pt2:pt2+leng]
#     # roi = cv2.GaussianBlur(roi, (5, 5), 0)
#     # roi = cv2.addWeighted(roi, 2.5, roi, -0.5, 0)
#     ret, roi = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY)
#     # Resize the image
#     roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
#     # roi = cv2.dilate(roi, (3, 3))
#     cv2.imshow('canvas', roi)
#     cv2.waitKey()
#     nbr = model.predict(np.array([roi], 'float64'))
#     # Calculate the HOG features
#     # roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(
#     # 14, 14), cells_per_block=(1, 1), visualize=False)
#     # nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
#     print(nbr)
#     cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),
#                 cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
