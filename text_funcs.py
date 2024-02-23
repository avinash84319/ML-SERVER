# all imports
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import logging

# Set up logging
logging.basicConfig(filename='error.log', level=logging.ERROR)

try:
    # load the models from disk
    action_model = tf.keras.models.load_model('./Models/action.keras')
    location_model = tf.keras.models.load_model('./Models/location.keras')
    when_model = tf.keras.models.load_model('./Models/when.keras')
    disease_model = tf.keras.models.load_model('./Models/disease.keras')
except Exception as e:
    logging.error("Error loading models from disk: " + str(e))

try:
    # load the tokenizers from disk
    with open('./Tokenizers/action_tokenizer.pkl', 'rb') as handle:
        action_tokenizer = pickle.load(handle)

    with open('./Tokenizers/location_tokenizer.pickle', 'rb') as handle:
        location_tokenizer = pickle.load(handle)

    with open('./Tokenizers/when_tokenizer.pickle', 'rb') as handle:
        when_tokenizer = pickle.load(handle)

    with open('./Tokenizers/disease_tokenizer.pickle', 'rb') as handle:
        disease_tokenizer = pickle.load(handle)
except Exception as e:
    logging.error("Error loading tokenizers from disk: " + str(e))

try:
    # load the label encoders from disk
    with open('./Label_encoders/action_label_encoder.pkl', 'rb') as handle:
        action_label_encoder = pickle.load(handle)

    with open('./Label_encoders/location_label_encoder.pickle', 'rb') as handle:
        location_label_encoder = pickle.load(handle)

    with open('./Label_encoders/when_label_encoder.pickle', 'rb') as handle:
        when_label_encoder = pickle.load(handle)

    with open('./Label_encoders/disease_label_encoder.pickle', 'rb') as handle:
        disease_label_encoder = pickle.load(handle)
except Exception as e:
    logging.error("Error loading label encoders from disk: " + str(e))


# import disease data

disease_data = pd.read_csv('./data/Disease_DOC_Type.csv')

def get_action(new_queries):

    new_sequences = action_tokenizer.texts_to_sequences(new_queries)
    new_padded_sequences = pad_sequences(new_sequences, maxlen=16)       # maxlen is the max length of the sequence but make it also automatic find a way
    predictions = action_model.predict(new_padded_sequences)
    predicted_classes = [action_label_encoder.classes_[tf.argmax(prediction).numpy()] for prediction in predictions]

    return predicted_classes

def get_location(new_queries):

    new_sequences = location_tokenizer.texts_to_sequences(new_queries)
    new_padded_sequences = pad_sequences(new_sequences, maxlen=26)       # maxlen is the max length of the sequence but make it also automatic find a way
    predictions = location_model.predict(new_padded_sequences)
    predicted_classes = [location_label_encoder.classes_[tf.argmax(prediction).numpy()] for prediction in predictions]

    return predicted_classes

def get_when(new_queries):

    new_sequences = when_tokenizer.texts_to_sequences(new_queries)
    new_padded_sequences = pad_sequences(new_sequences, maxlen=32)       # maxlen is the max length of the sequence but make it also automatic find a way
    predictions = when_model.predict(new_padded_sequences)
    predicted_classes = [when_label_encoder.classes_[tf.argmax(prediction).numpy()] for prediction in predictions]

    return predicted_classes


def get_disease(new_queries):

    new_sequences = disease_tokenizer.texts_to_sequences(new_queries)
    new_padded_sequences = pad_sequences(new_sequences, maxlen=78)       # maxlen is the max length of the sequence but make it also automatic find a way
    predictions = disease_model.predict(new_padded_sequences)
    predicted_classes = [disease_label_encoder.classes_[tf.argmax(prediction).numpy()] for prediction in predictions]
    predicted_classes = list(predicted_classes)
    for i in range(len(predicted_classes)):
        predicted_classes[i] = int(predicted_classes[i])-1               # subtract 1 from each value to get the correct index
    predicted_classes = disease_data.iloc[predicted_classes]
    predicted_classes = predicted_classes.to_dict(orient='records')
    return predicted_classes

def get_disease_constraints(new_queries):

    new_sequences = disease_tokenizer.texts_to_sequences(new_queries)
    new_padded_sequences = pad_sequences(new_sequences, maxlen=78)       # maxlen is the max length of the sequence but make it also automatic find a way
    predictions = disease_model.predict(new_padded_sequences)
    predicted_classes = [disease_label_encoder.classes_[tf.argmax(prediction).numpy()] for prediction in predictions]
    predicted_classes = list(predicted_classes)
    return predicted_classes