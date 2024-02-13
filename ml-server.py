from flask import Flask, request, jsonify
import logging
from text_funcs import get_action, get_location, get_when, get_disease
from rec_funcs import get_bestdocs
from time_funcs import get_time


# Set up logging
logging.basicConfig(filename='error.log', level=logging.ERROR)

app = Flask(__name__)


@app.route('/')
def index():
    try:
        return 'Hello Welcome to the ML Server for the super search', 200
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
        bestdocs = get_bestdocs(data['patient_info'])

        return jsonify({'bestdocs': bestdocs}), 200
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
