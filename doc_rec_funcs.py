# all imports
import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
import logging
import requests

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
    final_doctors_list=final_doctors_list[final_doctors_list['Doctor Location']<25]
    final_doctors_list=final_doctors_list['Doctor ID'].values
    final_doctors_list=list(final_doctors_list)
    for i in range(len(final_doctors_list)):
        final_doctors_list[i]=int(final_doctors_list[i])

    return final_doctors_list


def get_bestdocs_constraints(patient_info, constraints , next_req):
    try:
        # hospital for 0 and direct doctor for 1
        if next_req == 1:
            data_doctors = requests.post('http://host.internal.docker:3000/api/doctor/doctorRecommenderSystemInfo', json={'constraints': constraints})['data']
        else:
            data_doctors = requests.post('http://host.internal.docker:8000/api/hospital/doctorRecommenderSystemInfo', json={'constraints': constraints})['data']
        
        data_patients = [patient_info for i in range(len(data_doctors))]
        data_patients = np.array(data_patients)
        ypred = recommender_model.predict([data_patients, data_doctors])
        ypred = np.argmax(ypred, axis=1)
        sorted_index = np.argsort(ypred)
        final_doctors_list = data_doctors.iloc[sorted_index]
        final_doctors_list = final_doctors_list[final_doctors_list['Doctor Location'] < 25]
        final_doctors_list = final_doctors_list['Doctor ID'].values
        final_doctors_list = list(final_doctors_list)
        for i in range(len(final_doctors_list)):
            final_doctors_list[i] = int(final_doctors_list[i])

        return final_doctors_list
    except Exception as e:
        logging.error("Error in get_bestdocs_constraints: " + str(e))
