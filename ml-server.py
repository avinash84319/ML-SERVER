from flask import Flask, request, jsonify
import pandas as pd
import logging
from text_funcs import get_action, get_location, get_when, get_disease
from doc_rec_funcs import get_bestdocs,get_bestdocs_constraints
from hos_rec_funcs import get_besthos,get_besthos_constraints
from time_funcs import get_time


# Set up logging
logging.basicConfig(filename='error.log', level=logging.ERROR)

app = Flask(__name__)

# data import for mapping disease type to docter type
dis_doc_map=pd.read_csv('./data/Disease_DOC_Type.csv')

@app.route('/')
def index():
    try:
        return 'Hello Welcome to the ML Server for ', 200
    except Exception as e:
        logging.error("Error in index(): " + str(e))
        return 'Internal Server Error', 500

@app.route('/text', methods=['POST'])
def text():
    try:
        data = request.get_json()
        action = get_action(data['text'])
        location = get_location(data['text'])
        when = get_when(data['text'])
        print(action, location, when)

        return jsonify({'action': action, 'location': location, 'when': when}), 200
    except Exception as e:
        logging.error("Error in text(): " + str(e))
        return 'Internal Server Error', 500

@app.route('/disease', methods=['POST'])
def disease():
    try:
        data = request.get_json()
        disease = get_disease(data['text'])

        return jsonify({'Cases': disease}), 200
    except Exception as e:
        logging.error("Error in disease(): " + str(e))
        return 'Internal Server Error', 500


@app.route('/bestdocs', methods=['POST'])
def bestdocs():
    try:
        data = request.get_json()
        # patient info is Patient ID,Patient Age,Patient Gender,Patient Health Condition,Previous Doctor Type,Last Appointment (Days)
        bestdocs = get_bestdocs(data['patient_info'])

        return jsonify({'bestdocs': bestdocs}), 200
    except Exception as e:
        logging.error("Error in bestdocs(): " + str(e))
        return 'Internal Server Error', 500

@app.route('/besthos', methods=['POST'])
def besthos():
    try:
        data = request.get_json()
        besthos = get_besthos(data['patient_info'])

        return jsonify({'besthos': besthos}), 200
    except Exception as e:
        logging.error("Error in bestdocs(): " + str(e))
        return 'Internal Server Error', 500


@app.route('/timepred', methods=['POST'])
def timepred():
    try:
        data=request.get_json()
        time=get_time(data['info'])

        return jsonify({'time_mins':time}), 200
    except Exception as e:
        logging.error("Error in timepred(): " + str(e))
        return 'Internal Server Error', 500

@app.route('/mlapi/text', methods=['POST'])
def mlapitext():
    try:
        data = request.get_json()
        action = get_action(data['text'])
        location = get_location(data['text'])
        when = get_when(data['text'])
        print(action, location, when)

        return jsonify({'action': action, 'location': location, 'when': when}), 200
    except Exception as e:
        logging.error("Error in text(): " + str(e))
        return 'Internal Server Error', 500

@app.route('/mlapi/disease', methods=['POST'])
def mlapidisease():
    try:
        data = request.get_json()
        disease = get_disease(data['text'])
        patient_info=data['patient_info']
        constraints=data['constraints']
        next_req=data['next_req']

        # upade patient info based on disease assuming last doctor type and patient health condition to based on disease
        for i in range(len(patient_info)):
            patient_info[i][3]=int(disease[i])
            patient_info[i][4]=int(dis_doc_map[dis_doc_map['Disease']==disease[i]]['Doctor_Type_ID'].values[0])

        bestdocs =  []
        for i in range(len(patient_info)):
            bestdocs.append(get_bestdocs_constraints(patient_info[i],constraints[i],next_req[i]))

        return jsonify({'Cases': disease,'Docs_for_Cases':bestdocs}), 200
    except Exception as e:
        logging.error("Error in disease(): " + str(e))
        return 'Internal Server Error', 500


@app.route('/mlapi/bestdocs', methods=['POST'])
def mlapibestdocs():
    try:
        data = request.get_json()
        constraints=data['constraints']
        next_req=data['next_req']
        # patient info is Patient ID,Patient Age,Patient Gender,Patient Health Condition,Previous Doctor Type,Last Appointment (Days)
        bestdocs = []
        for i in range(len(data['patient_info'])):
            bestdocs.append(get_bestdocs_constraints(data['patient_info'][i],constraints[i],next_req[i]))

        return jsonify({'bestdocs': bestdocs}), 200
    except Exception as e:
        logging.error("Error in bestdocs(): " + str(e))
        return 'Internal Server Error', 500

@app.route('/mlapi/besthos', methods=['POST'])
def mlapibesthos():
    try:
        data = request.get_json()
        constraints=data['constraints']

        besthos = []
        for i in range(len(data['patient_info'])):
            besthos.append(get_besthos_constraints(data['patient_info'],constraints[i]))

        return jsonify({'besthos': besthos}), 200
    except Exception as e:
        logging.error("Error in bestdocs(): " + str(e))
        return 'Internal Server Error', 500


@app.route('/mlapi/timepred', methods=['POST'])
def mlapitimepred():
    try:
        data=request.get_json()
        time=get_time(data['info'])

        return jsonify({'time_mins':time}), 200
    except Exception as e:
        logging.error("Error in timepred(): " + str(e))
        return 'Internal Server Error', 500


if __name__ == '__main__':
    app.run(debug=True, port=7000)
