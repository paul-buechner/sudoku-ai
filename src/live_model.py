import numpy as np
import cv2
import math
import imutils
import tensorflow as tf

from scipy import ndimage
from tensorflow import keras
from image_search import processing
from model_prediction import model_prediction


# Testing Webcam
'''
cams_test = 10
for i in range(-1, cams_test):
    cap = cv2.VideoCapture(i)
    test, frame = cap.read()
    print("i : "+str(i)+" /// result: "+str(test))
'''

# Variables

width = 640
height = 480
threshold = 80
image_dimensions = (28, 28, 3)
padding_row = 10
padding_col = 10


cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# !! Load Model !!
model = tf.keras.models.load_model('model/model_tf.h5')


def get_image_contour_thresh(image):
    h_image, w_imgage, _ = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # operations to cleanup the thresholded image
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 33, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 4))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # find contours in the thresholded image, then initialize the
    # digit contours lists
    invert = 255 - thresh
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    digitCnts = []
    # loop over the digit area candidates
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        # if the contour is sufficiently large, it must be a digit
        if (w >= 5 and w <= 50) and (h >= 35 and h <= 100) and (w/h >= (0.375) and w/h <= 1):
            digitCnts.append(c)
            cv2.rectangle(image, (x, y),
                          (x + w, y + h), (50, 50, 255), 2)
    return image, digitCnts, thresh, gray


# Webcam Config
while True:
    success, image_original = cap.read()

    # !! Dynamic digit recognition and cropping !!
    image, digitCnts, thresh, gray = get_image_contour_thresh(image_original)
    for c in digitCnts:
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)
        # Extract out the object and place into output image
        image = np.zeros_like(gray)
        image[mask >= 1] = 255

        # Cropping...
        # image = cv2.bitwise_and(gray, mask)
        # cv2.imshow("mask", image)

        (y, x) = np.where(mask == 255)
        (top_y, top_x) = (np.min(y), np.min(x))
        (bottom_y, bottom_x) = (np.max(y), np.max(x))
        image_crop = image[top_y - 5:bottom_y + 5, top_x - 5:bottom_x + 5]
        image_crop = cv2.bitwise_not(image_crop)
        image_crop = cv2.equalizeHist(image_crop)
        # Processing for neural network

        if image_crop is not None:
            image_processed, state = processing(image_crop)
    # '''
        # !! Prediction !!
        # Skip manually by boday ratio
        if state == True:
            class_index, prob_value = 1, 1

        # Predict image with AI-Model
        elif (image_processed is not None) and (state == False):
            class_index, prob_value = model_prediction(image_processed, model)

        elif image_processed is None:
            class_index, prob_value = 0, 1

        else:
            class_index, prob_value = 0, 0

        prob_value = (prob_value*100)
        print(class_index, prob_value)

    if prob_value > threshold:
        cv2.putText(image_original, str(class_index) + "  " + str((prob_value)),
                    (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    # '''
    cv2.imshow("Original Image: ", image_original)
    # cv2.imshow("Processing Image: ", thresh)

    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
