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
# Read the input image
n = 0
while n <= 4:
    # n = input('Enter maze no:')
    print('For Image: ' + str(n))
    im = cv2.imread("maze0"+str(n)+".jpg")

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
        0: [(44, 44, 4), (164, 44, 0), (164, 124, 6),
            (284, 124, 7), (164, 204, 7), (84, 324, 2)],
        1: [(44, 164, 2), (44, 204, 6), (284, 284, 4)],
        2: [(364, 324, 0), (324, 44, 9), (204, 124, 3),
            (84, 124, 1), (44, 324, 7)],
        3: [(44, 44, 4), (164, 124, 6), (284, 123, 7),
            (44, 324, 0), (284, 364, 8)],
        4: [(4, 124, 9), (84, 84, 6), (324, 84, 0),
            (324, 164, 1), (244, 164, 6), (244, 324, 1), (164, 364, 2), (4, 364, 3)]
    }

    # print(rects)
    # For each rectangular region, calculate HOG features and predict
    # the digit using Linear SVM.

    # test_loss, test_accuracy = test.evaluate(test_images, test_labels)
    # print(test_accuracy)
    i = 0
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
        ret, roi = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY_INV)
        # Resize the image
        # print(roi.shape)
        # cv2.imshow('canvas0', roi)
        # cv2.waitKey()
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (1, 1))
        roi = roi.astype('float32') / 255
        # cv2.imshow('canvas1', roi)
        # cv2.waitKey()
        # Calculate the HOG features
        # roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(
        #     14, 14), cells_per_block=(1, 1), visualize=False)
        rroi = np.array(roi).reshape(28, 28, 1)
        nbr = test.predict_classes(np.array([rroi], 'float64'))
        print(rect[2], nbr)
        if(rect[2] == nbr[0]):
            i = i + 1
    print("predicted " + str(i) + "/" + str(len(rects[int(n)])) + " correct")
    n = n+1
    # cv2.imshow("Resulting Image with Rectangular ROIs", im)
    # cv2.waitKey()

cv2.destroyAllWindows()
