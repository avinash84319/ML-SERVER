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
    data_hospital=pd.read_csv('./data/hospital_data.csv')
except Exception as e:
    logging.error("Error loading data from disk: " + str(e))


try:
    # load the models from disk
    recommender_model=tf.keras.models.load_model('./Models/hos-recommender-ensemble.h5')
except Exception as e:
    logging.error("Error loading models from disk: " + str(e))


def get_besthos(patient_info):

    data_patients=[patient_info for i in range(len(data_hospital))]
    data_patients=np.array(data_patients)
    ypred=recommender_model.predict([data_patients,data_hospital])
    ypred=np.argmax(ypred,axis=1)
    sorted_index=np.argsort(ypred)
    final_hospital_list=data_hospital.iloc[sorted_index]
    final_hospital_list=final_hospital_list[final_hospital_list['hospital Location']<25]
    final_hospital_list=final_hospital_list['hospital ID'].values
    final_hospital_list=list(final_hospital_list)
    for i in range(len(final_hospital_list)):
        final_hospital_list[i]=int(final_hospital_list[i])

    return final_hospital_list

def get_besthos_constraints(patient_info,constraints):

    # get hospitals based on constraints
    data_hospital=requests.post('http://localhost:3000/api/doctor/doctorRecommenderSystemInfo', json={'constraints': constraints})['data']
    
    data_patients=[patient_info for i in range(len(data_hospital))]
    data_patients=np.array(data_patients)
    ypred=recommender_model.predict([data_patients,data_hospital])
    ypred=np.argmax(ypred,axis=1)
    sorted_index=np.argsort(ypred)
    final_hospital_list=data_hospital.iloc[sorted_index]
    final_hospital_list=final_hospital_list[final_hospital_list['hospital Location']<25]
    final_hospital_list=final_hospital_list['hospital ID'].values
    final_hospital_list=list(final_hospital_list)
    for i in range(len(final_hospital_list)):
        final_hospital_list[i]=int(final_hospital_list[i])

    return final_hospital_list