from instance import LATINInstance
from flask import Flask, request
from flask_cors import CORS
import logging
import random
import yaml

def load_settings(path="settings.yaml"):
    with open(path, "r") as file:
        settings = yaml.safe_load(file)
    return settings

# Beispielhafte Nutzung
config = load_settings()
#if config.get("CreateInstanceForEveryIp", False):
#   print("Instanz wird pro IP erstellt.")


app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)

if config.get("CreateInstanceForEveryIp",True):
    AssignedInstances = {} 
else:
    StandardInstance = LATINInstance(False)
    training_data = StandardInstance.processTrainingData()
    StandardInstance.prepareInstance(training_data)
@app.route("/ai", methods=["GET", "POST"])
def ai():
    user_ip = request.headers.get('X-Forwarded-For', request.remote_addr).split(',')[0]
    data = request.get_json(silent=True) or {}
    question = data.get("question", request.args.get("question", "Keine Frage übermittelt"))

    if user_ip not in AssignedInstances:
        if config.get("CreateInstanceForEveryIp",True):
            if config.get("DebugMode",False):
                logging.info(f"Neue IP erkannt: {user_ip}")
            instance = LATINInstance(False)
            training_data = instance.processTrainingData()
            instance.prepareInstance(training_data)
            AssignedInstances[user_ip] = instance
            response, NLPInfo = instance.chatbot_response(question)
            return {"response": response, "info": NLPInfo}
        else:
            response, NLPInfo = StandardInstance.chatbot_response(question)
            return {"response": response, "info": NLPInfo}
    else:
        if config.get("CreateInstanceForEveryIp", True):
            if config.get("DebugMode", False):
                logging.info(f"Bekannte IP: {user_ip}")
            instance = AssignedInstances[user_ip]
            response, NLPInfo = instance.chatbot_response(question)
            return {"response": response, "info": NLPInfo}
        else:
            response, NLPInfo = StandardInstance.chatbot_response(question)
            return {"response": response, "info": NLPInfo}


if __name__ == '__main__':
    app.run(port=5001)