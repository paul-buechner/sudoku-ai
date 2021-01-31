import numpy as np
import cv2
import imutils
import argparse
import os
from tensorflow import keras
from imutils import contours
from skimage.filters import threshold_local

from src.sudoku import *
from src.image_search import processing, sort
from src.model_prediction import model_prediction


def main():
    # !! Select image path !!
    path = "assets/samples/sample1.jpg"

    # !! Load Model !!
    model = keras.models.load_model('model/model_tf.h5')

    # Load board
    board = grid_operator(path, model)

    # Solve it...
    test()
    solve_all([(board)], None, 0.0)


# Generate grid with given numers using AI
def grid_operator(image, model):
    # Filter contours and fix lines, output --> grid
    sudoku_rows, row, gray = sort(image)

    # Define local variables
    board, pos_iD = [], 0
    for row in sudoku_rows:
        for c in row:
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)

            # Extract out the object and place into output image
            image = np.zeros_like(gray)
            image[mask == 0] = 255

            # Cropping...
            image = cv2.bitwise_and(gray, mask)
            '''cv2.imshow("mask", image)
            cv2.waitKey(300)'''

            (y, x) = np.where(mask == 255)
            (top_y, top_x) = (np.min(y), np.min(x))
            (bottom_y, bottom_x) = (np.max(y), np.max(x))
            # +6, -5 horizontel Cutoff
            image_crop = image[top_y + 4:bottom_y - 4, top_x + 5:bottom_x - 5]

            # Process image for further operations
            image_processed, state = processing(image_crop)

            # Skip manually by boday ratio:
            if state == True:
                board.append(str(1))

            # Predict image with AI-Model
            elif (image_processed is not None) and (state == False):
                class_index, prob_value = model_prediction(
                    image_processed, model)
                '''print("Predicted Class: ", class_index,
                      "Prob_Val: ", prob_value)'''

                # Add predicted class to board
                board.append(str(class_index))

            elif image_processed is None:
                board.append(str(0))

    return ''.join(board)


if __name__ == '__main__':
    main()
