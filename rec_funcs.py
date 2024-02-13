# all imports
import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
import logging

# Set up logging
logging.basicConfig(filename='error.log', level=logging.ERROR)

try:
    # load the data from disk
    data_doctors=pd.read_csv('./data/doctors_data.csv')
except Exception as e:
    logging.error("Error loading data from disk: " + str(e))


try:
    # load the models from disk
    recommender_model=tf.keras.models.load_model('./Models/recommender-ensemble.h5')
except Exception as e:
    logging.error("Error loading models from disk: " + str(e))


def get_bestdocs(patient_info):

    data_patients=[patient_info for i in range(len(data_doctors))]
    data_patients=np.array(data_patients)
    ypred=recommender_model.predict([data_patients,data_doctors])
    ypred=np.argmax(ypred,axis=1)
    sorted_index=np.argsort(ypred)
    final_doctors_list=data_doctors.iloc[sorted_index]
    final_doctors_list=final_doctors_list[final_doctors_list['Doctor Location']==patient_info[2]]
    final_doctors_list=final_doctors_list['Doctor ID'].values
    final_doctors_list=list(final_doctors_list)
    for i in range(len(final_doctors_list)):
        final_doctors_list[i]=int(final_doctors_list[i])

    return final_doctors_list