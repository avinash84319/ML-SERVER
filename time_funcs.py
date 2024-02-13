# all imports
import tensorflow as tf
import logging

# Set up logging
logging.basicConfig(filename='error.log', level=logging.ERROR)

try:
    # load the models from disk
    time_model=tf.keras.models.load_model('./Models/time.keras')
except Exception as e:
    logging.error("Error loading models from disk: " + str(e))

def get_time(info):

    predictions=time_model.predict(info)
    predictions=list(predictions)
    for i in range(len(predictions)):
        predictions[i]=int(predictions[i])

    return predictions