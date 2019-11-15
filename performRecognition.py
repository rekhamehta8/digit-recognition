# Import the modules
import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np

# Load the classifier
clf = joblib.load("digits_cls.pkl")

# Read the input image
n = input('Enter maze no:')
im = cv2.imread("maze0"+n+".jpg")

# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

# Threshold the image
# ret, im_th = cv2.threshold(im_gray, 200, 255, cv2.THRESH_BINARY)
# cv2.imshow('canvas', im_th)
# cv2.waitKey()

# # Find contours in the image
# ctrs, hier = cv2.findContours(
#     im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # Get rectangles contains each contour
# rects = [cv2.boundingRect(ctr) for ctr in ctrs]
rects = {
    0: [(44, 44, 32, 32), (164, 44, 32, 32), (164, 124, 32, 32),
        (284, 124, 32, 32), (164, 204, 32, 32), (84, 324, 32, 32)],
    1: [(44, 164, 40, 40), (44, 204, 40, 40), (284, 284, 40, 40)],
    2: [(364, 324, 40, 40), (324, 44, 40, 40), (204, 124, 40, 40),
        (84, 124, 40, 40), (44, 324, 40, 40)],
    3: [(44, 44, 40, 40), (164, 124, 40, 40), (284, 123, 40, 40),
        (44, 324, 40, 40), (284, 364, 40, 40)],
    4: [(4, 124, 40, 40), (84, 84, 40, 40), (324, 84, 40, 40),
        (324, 164, 40, 40), (244, 164, 40, 40), (244, 324, 40, 40), (164, 364, 40, 40), (4, 364, 40, 40)]
}
# print(rects)
# For each rectangular region, calculate HOG features and predict
# the digit using Linear SVM.
for rect in rects[int(n)]:
    # Draw the rectangles
    # cv2.rectangle(im, (rect[0], rect[1]), (rect[0] +
    #                                        rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
    # Make the rectangular region around the digit
    leng = 32
    pt1 = int(rect[1])
    pt2 = int(rect[0])
    # roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
    roi = im_gray[pt1:pt1+leng, pt2:pt2+leng]
    # roi = cv2.GaussianBlur(roi, (5, 5), 0)
    # roi = cv2.addWeighted(roi, 2.5, roi, -0.5, 0)
    ret, roi = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY)
    # Resize the image
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))
    cv2.imshow('canvas', roi)
    cv2.waitKey()
    # Calculate the HOG features
    roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(
        14, 14), cells_per_block=(1, 1), visualize=False)
    nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
    print(nbr)
    cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),
                cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.waitKey()
cv2.destroyAllWindows()
