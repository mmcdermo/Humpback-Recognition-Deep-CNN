"""
 server.py
 Runs a simple API granting access to all available experiments.
 Usage: python server.py

 Environmental parameters for each experiment should be stored in <ExperimentName>_params.json

 This server uses a separate process to perform predictions, with a pipe from `multiprocessing` connecting each flask server thread to the prediction process. Access to the server side of the pipe is guarded by a Lock from multiprocessing. 
"""

# We must import skimage.io before we import any other imaging libraries,
#  or PIL will be unable to load images for us.
#  This is a known bug in scikit image. 
import skimage.io

from flask import Flask, jsonify, request, abort, make_response
import experiments
import manage
import json
from multiprocessing import Queue, Process, Pipe, Lock
import multiprocessing
import Queue as Q
import random
import os.path

app = Flask(__name__)

@app.route('/algorithms/')
def algorithms():
    experimentDict = manage.experimentClasses()
    print(experimentDict.keys())
    return jsonify({"algorithms": experimentDict.keys()})

@app.route('/algorithm/<string:algorithmName>/predictImage', methods=['GET'])
def predictImage(algorithmName):
    # Ensure filename is provided correctly
    filename = request.args.get('filename', '')
    if filename == "":
        return make_response(jsonify({'error': 'Filename not provided'}), 400)
    if not os.path.isfile(filename):
        print("Error: File "+filename+" not found")
        return make_response(jsonify({'error': 'Could not find file '+filename}), 400)
    # Ensure experiment exists
    experimentDict = manage.experimentClasses()
    if algorithmName not in experimentDict:
        return make_response(jsonify({'error': 'Experiment does not exist'}), 404)

    # Load environment parameters from file
    envParams = {}    
    try:    
        with open(algorithmName+"_params.json", 'r') as file:
            envParams = json.loads(file.read())
    except:
        return make_response(jsonify({'error': 'Failed to load environmental parameters from '+algorithmName+"_params.json"}), 400)        

    # Send our task over the pipe and wait for a result. 
    resId = str(random.random())
    serverPipeLock.acquire()
    serverPipe.send({"algorithmName": algorithmName,
                   "filename": filename,
                   "envParams": envParams,
                   "resId": resId
    })
    res = serverPipe.recv()
    serverPipeLock.release()

    return jsonify(res)
    
def runPredictions(pipe):
    experimentDict = manage.experimentClasses()
    while True:
        try:
            item = pipe.recv()
            experiment = experimentDict[item["algorithmName"]]()
            experiment.envParams = item["envParams"]
            try:
                predictions = experiment.predictImage(item["filename"])
                pipe.send({"predictions": predictions})
            except Exception as e:
                pipe.send({"error": "Exception: "+str(e)})
        except Q.Empty:
            print("Timed out retrieving item from queue")
    

serverPipe, taskPipe = Pipe()
serverPipeLock = Lock()

if __name__ == '__main__':
    # Startup a task process to process predictions sequentially
    taskProcess = Process(target=runPredictions, args=((taskPipe,)))
    taskProcess.start()
    app.run(host='0.0.0.0', debug=True)
