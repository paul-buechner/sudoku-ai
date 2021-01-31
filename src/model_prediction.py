import numpy as np


# Predict image
def model_prediction(image_processed, model):
    predictions = np.array(model.predict_on_batch(image_processed))
    prob_value = np.amax(predictions)
    class_index = int(
        np.argmax(model.predict_on_batch(image_processed), axis=-1))
    return class_index, prob_value
