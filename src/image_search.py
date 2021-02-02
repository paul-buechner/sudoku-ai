import numpy as np
import cv2
import imutils
import argparse
import os

from imutils import contours
from skimage.filters import threshold_local


def order_points(pts):
    '''
    initialzie a list of coordinates that will be ordered
    such that the first entry in the list is the top-left,
    the second entry is the top-right, the third is the
    bottom-right, and the fourth is the bottom-left
    '''
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    '''
    now, compute the difference between the points, the
    top-right point will have the smallest difference,
    whereas the bottom-left will have the largest difference
    '''
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    '''
    compute the width of the new image, which will be the
    maximum distance between bottom-right and bottom-left
    x-coordiates or the top-right and top-left x-coordinates
    '''
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    '''
    compute the height of the new image, which will be the
    maximum distance between the top-right and bottom-right
    y-coordinates or the top-left and bottom-left y-coordinates
    '''
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    '''
    now that we have the dimensions of the new image, construct
    the set of destination points to obtain a "birds eye view",
    (i.e. top-down view) of the image, again specifying points
    in the top-left, top-right, bottom-right, and bottom-left
    order
    '''
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


# construct the argument parser and parse the arguments
'''
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to the image to be scanned")
args = vars(ap.parse_args())
'''


def scan(image):
    # load the image and compute the ratio of the old height
    # to the new height, clone it, and resize it
    image = cv2.imread(image)
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)

    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # outline with white pixelborder
    '''gray = np.pad(gray, ((2, 2), (2, 2)), "constant", constant_values=255)'''

    edged = cv2.Canny(gray, 75, 160)  # 200 default for y

    # show the original image and the edge detected image
    '''
    print("STEP 1: Edge Detection")
    cv2.imshow("Image", image)
    cv2.imwrite("assets/processing/image.jpg", image)

    cv2.imshow("Edged", edged)
    cv2.imwrite("assets/processing/edged.jpg", edged)

    cv2.imshow("Blur", gray)
    cv2.imwrite("assets/processing/blur.jpg", gray)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break

    # show the contour (outline) of the piece of paper
    '''
    print("STEP 2: Find contours of paper")
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)

    cv2.imshow("Outline", image)
    cv2.imwrite("assets/processing/outline.jpg", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

    # apply the four point transform to obtain a top-down
    # view of the original image
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    warped = imutils.resize(warped, height=650)

    # convert the warped image to grayscale, then threshold it
    # to give it that 'black and white' paper effect
    '''
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warped, 81, offset=10, method="gaussian")
    warped = (warped > T).astype("uint8") * 255
    # th3 = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                            cv2.THRESH_BINARY_INV, 11, 2)

    '''
    # show the original and scanned images
    '''
    print("STEP 3: Apply perspective transform")

    cv2.imshow("Original", imutils.resize(orig, height=650))

    cv2.imshow("Scanned", imutils.resize(warped, height=650))
    cv2.imwrite("assets/processing/warped.jpg", warped)

    # cv2.imshow("TH3", imutils.resize(th3, height=650))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

    return warped


# Process single image
def processing(image):
    state = False
    IMAGE_DIMENSIONS = (28, 28, 3)

    ret, thresh = cv2.threshold(image.copy(), 87, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.imwrite("assets/processing/processing_step1.jpg", image)

    if contours == []:
        return None, state

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # Creating a rectangle around the digit in the original image (for displaying the digits fetched via contours)
        cv2.rectangle(image, (x, y), (x+w, y+h),
                      color=(0, 255, 0), thickness=2)

        # Cropping out the digit from the image corresponding to the current contours in the for loop
        digit = thresh[y:y+h, x:x+w]
        cv2.imwrite("assets/processing/processing_step2.jpg", digit)

        # Body ratio, identifier between 1 and 7
        if (w/h) <= 0.45:
            state = True

        # Resizing that digit to (18, 18)
        resized_digit = cv2.resize(digit, (14, 18))

        # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
        padded_digit = np.pad(resized_digit, ((5, 5), (7, 7)),
                              "constant", constant_values=0)

    cv2.imwrite("assets/processing/processing_step3.jpg", padded_digit)
    processed_image = padded_digit.reshape(
        1, IMAGE_DIMENSIONS[0], IMAGE_DIMENSIONS[1], 1)

    return processed_image, state


# Filter contours and fix lines, output --> grid
def sort(image):
    # Load image, grayscale, and adaptive threshold
    image = scan(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 33, 5)
    '''
    cv2.imshow("thresh", thresh)
    cv2.imwrite("assets/processing/thresh.jpg", thresh)

    cv2.waitKey()
    cv2.destroyAllWindows()
    '''

    # Filter out all numbers and noise to isolate only boxes
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 1200:
            cv2.drawContours(thresh, [c], -1, (0, 0, 0), -1)

    # Fix horizontal and vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                              vertical_kernel, iterations=9)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                              horizontal_kernel, iterations=4)

    # Sort by top to bottom and each row by left to right
    invert = 255 - thresh
    '''
    cv2.imshow("invert", invert)
    cv2.imwrite("assets/processing/invert.jpg", invert)

    cv2.waitKey()
    cv2.destroyAllWindows()
    '''

    cnts = cv2.findContours(invert, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    (cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")

    sudoku_rows = []
    row = []
    for (i, c) in enumerate(cnts, 1):
        area = cv2.contourArea(c)
        if area < 50000:
            row.append(c)
            if i % 9 == 0:
                (cnts, _) = contours.sort_contours(row, method="left-to-right")
                sudoku_rows.append(cnts)
                row = []

    return sudoku_rows, row, gray
