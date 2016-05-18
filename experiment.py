import h5py
import os
import time
import util
import math
import json
import hashlib
from keras import backend as K

def envParam(name, desc="", required=False):
    return {"name": name, "desc": desc, "required": required}

class MissingParameter(Exception):
        pass
    
class Experiment():
    """
    The Experiment class provides a way to track model performance over
    numerous trials with varying parameters, and to reproduce those trials.

    The state of the underlying model is saved throughout the trial along with
    performance measures, so that the model can be observed during training
    (at each epoch, for example).
    """
    
    params = {}
    model = {}
    numTrials = 0
    experimentName="defaultExperiment"
    
    def __init__(self):
        pass

    @staticmethod
    def listEnvParams(self):
        """
        Override this function to list environment parameters.
        These will be available via self.params after initialization.
        """
        return [
            envParam("note", desc="This experiment has no custom environment parameters")]

    def experimentDir(self):
        rootDir = "experiments/"
        if "experimentDir" in self.envParams and self.envParams["experimentDir"]:
            rootDir = self.envParams["experimentDir"]
        return rootDir + self.experimentName

    def run(self, envParams):
        """
        Run the experiment.

        One trial will be run for each provided parameter combination produced by
        `genTrialParams`
        """
        # Ensure all required environment parameters are provided
        for param in self.listEnvParams():
            if param["name"] not in envParams and param["required"]:
                raise MissingParameter("Parameter " + param["name"] + "is required by this experiment")
        self.envParams = envParams

        # Generate trial parameters
        trialParams = self.genTrialParams()

        # Perform a trial for each unique trial parameter combination
        for paramSet in trialParams:
            print("Running trial with parameters: ")
            print(paramSet)
            self.runTrial(paramSet)

    def trialName(self, trialParams):
        """ Hash params to serve as the trial identifier """
        return 'trial_'+hashlib.md5(json.dumps(trialParams)).hexdigest()

    def getLastState(self, trialsFile, trialStr):
        maxState = -1
        if trialStr in trialsFile:
            for k in trialsFile[trialStr+"/states/"]:
                if int(k) > maxState:
                    maxState = int(k)

        return maxState

    def loadTrial(self, trialStr=None, stateNum=None):
        """ 
        Load the given state from a trial.
        If trialStr is not provided, it defaults to the first listed in the file.
        If stateNum is not provided, it defaults to the last state recorded.
        """
        trialsFile = h5py.File(self.experimentDir()+"/trials.h5")
        print("============================================")
        print(" LOADING TRIALS FROM "+self.experimentDir()+"/trials.h5")
        print("============================================")
        if trialStr == None:
            # Iterate until we can find a trial that has states
            for trial in trialsFile.keys():
                if trial+"/states" in trialsFile:
                    trialStr = trial
            if trialStr == None:
                raise Exception("No valid trials found")
            
        if stateNum == None:
            stateNum = self.getLastState(trialsFile, trialStr)
        trialParams = util.loadHDF5Dict(trialsFile, trialStr+"/parameters")
        
        lastState = trialsFile[trialStr+"/states/"+str(stateNum)+"/state"]
        self.setupTrial(trialParams)
        self.loadState(trialParams, lastState)
        trialsFile.close()
        return trialParams

    def runTrial(self, trialParams):
        """
        Builtin function to run a trial and record the results.
        """
        self.numTrials += 1

        # Perform trial
        try:
            os.makedirs(self.experimentDir())
        except:
            print("Experiment dir already exists: "+self.experimentDir())
        self.setupTrial(trialParams)

        trialStr = self.trialName(trialParams)
        trialsFile = h5py.File(self.experimentDir()+"/trials.h5", "a")
        if "experiment" not in trialsFile:
            trialsFile["experiment"] = self.experimentName

        maxState = self.getLastState(trialsFile, trialStr)

        # Load oldest max state
        if maxState != -1:
            print("Loading state from stored state: "+str(maxState))
            lastState = trialsFile[trialStr+"/states/"+str(maxState)+"/state"]
            self.loadState(trialParams, lastState)

        # Store each yielded state and performance measures
        stateNum = maxState + 1        
        for state in self.trial():
            print("Storing state "+str(stateNum)+" for trial "+trialStr)
            if trialStr+"/parameters" not in trialsFile:
                util.storeHDF5Dict(trialsFile, trialStr+"/parameters", trialParams)
            if trialStr+"/states/"+str(stateNum)+"/state" in trialsFile:
                print("State "+str(stateNum)+" already recorded for trial "+trialStr)
                exit(1)
            util.storeHDF5Dict(trialsFile, trialStr+"/states/"+str(stateNum)+"/state", state["state"])            
            util.storeHDF5Dict(trialsFile, trialStr+"/states/"+str(stateNum)+"/performance", state["performance"])
            util.storeHDF5Dict(trialsFile, trialStr+"/states/"+str(stateNum)+"/notes", state["notes"])
            stateNum += 1
        trialsFile.close()            

    def experimentSummary():
        """
        Summarize performance over all trials and states
        """
        trialsFile = h5py.File(self.experimentDir()+"/trials.h5", "a")
        measures = []
        minMeasures = {}
        maxMeasures = {}
        for trial in trialsFile:
            for state in trialsFile[trial+"/states/"]:
                for measure in state["performance"]:
                    m = state["performance"][measure]
                    stateName = trial + "/" + state
                    if measures == None:
                        measures.append(measure)
                        minMeasures[measure] = (stateName, m)
                        maxMeasures[measure] = (stateName, m)
                    else:
                        if m < minMeasures[m][1]:
                            minMeasures[m] = (stateName, m)
                        if m > maxMeasures[m][1]:
                            maxMeasures[m] = (stateName, m)
        for measure in measures:
            print("Measurement: "+measure)
            print("\tMin: "+minMeasures[measure][0]+" ("+minMeasures[measure][1]+")")
            print("\tMax: "+minMeasures[measure][0]+" ("+minMeasures[measure][1]+")")

        trialsFile.close()
        return (minMeasures, maxMeasures)

    def predict(X):
        """
        Predict output for a given dataset after a trial result has been loaded.
        """
        return self.model.predict(X)

    def evaluate(X):
        """
        Evaluate model performance on a dataset a trial result has been loaded.
        """
        return self.model.evaluate(X)

    def setupTrial(trialParams):
        """
        Override this function to setup your experiment before `trial` is run
        """
        pass
    
    def trial(trialParams):
        """
        Override this function to run a single trial of your experiment,
        given trialParams.
        
        runTrial should return yield state and performance measurements at points
        during the trial so that they can be reproduced. 
        """
        return [{"state": None, "performance": None, "notes": None}]

    def genTrialParams():
        """
        Override this function to return a list (or generator)
        of trial parameters for your experiment. Each element will
        spawn a new trial. 
        """
        return []

    def loadState(trialParams, state):
        """
        Override this function to load a given state into your model
        """
        return

def getKerasModelParameters(model):
    """
    Helper function to package a keras model into an object
    """
    f = {}
    layer_names = []
    for layer in model.layers:
        f[layer.name] = {}
        layer_names.append(layer.name)
        symbolic_weights = layer.trainable_weights + layer.non_trainable_weights
        weight_values = layer.get_weights()
        weight_names = []
        for i, (w, val) in enumerate(zip(symbolic_weights, weight_values)):
            if hasattr(w, 'name') and w.name:
                name = str(w.name)
            else:
                name = 'param_' + str(i)
            weight_names.append(name.encode('utf8'))
        f[layer.name]['weight_names'] = weight_names
        for name, val in zip(weight_names, weight_values):
            f[layer.name][name] = val
    f['layer_names'] = layer_names
    return f

def loadKerasModelParameters(model, params):
    """
    Helper function to load saved parameters into a keras model.
    Code adapted from keras internals.
    """
    layer_names = [n.decode('utf8') for n in params['layer_names']]
    if len(layer_names) != len(model.layers):
        raise Exception('Number of layers in `params` greater than number of layers in `model`')

    weight_value_tuples = []
    for k, name in enumerate(layer_names):
        g = params[name]
        weight_names = [n.decode('utf8') for n in g['weight_names']]
        if len(weight_names):
            weight_values = [g[weight_name] for weight_name in weight_names]
            layer = model.layers[k]
            symbolic_weights = layer.trainable_weights + layer.non_trainable_weights
            if len(weight_values) != len(symbolic_weights):
                raise Exception('Layer #' + str(k) + " has incorrect number of weights")
            weight_value_tuples += zip(symbolic_weights, weight_values)
            K.batch_set_value(weight_value_tuples)

    
class KerasExperiment(Experiment):
    """
    The KerasExperiment class extends Experiment to
    expediate development of keras models
    """
        
    def evaluate(filenames, X):
        # Run data through datagen, then evaluate        
        dataset = {}
        for idx in range(len(filenames)):
            dataset[filenames[idx]] = X[idx]

        for X_test, Y_test, filenames in self.datagen.generate(1, dataset=dataset):
            return self.model.evaluate(X_test, Y_test)
        
    def loadState(self, trialParams, state):
        loadKerasModelParameters(self.model.model, state)
        
    def trial(self):
        nEpochs = self.envParams["nEpochs"] if "nEpochs" in self.envParams else 200
        batchSize = 16
        if "batchSize" in self.envParams and self.envParams["batchSize"] != None:
            batchSize = self.envParams["batchSize"]
        for e in range(nEpochs):
            try:
                btime0 = time.time()
                print("=============================== Epoch "+str(e))

                # Evaluate classification performance
                print("Epoch pre-testing evaluation")
                for X_test, Y_test, filenames in self.datagen.generate(1, dataset=self.testingData):
                    self.model.evaluate(X_test, Y_test)

                batch = 0
                trainTime = 0
                time0 = time.time()
                losses = []

                nBatches = len(self.trainingData) / batchSize
                performance = {}
                for X_train, Y_train, filenames in self.datagen.generate(nBatches, dataset=self.trainingData):
                    batch += 1
                    trainTime0 = time.time()
                    #loss = model.trainOnBatch(X_train, Y_train)
                    loss = 0
                    trainTime += time.time() - trainTime0
                    losses.append(loss)
                
                    time1 = time.time()
                    if batch % 10 == 0:
                        print("Last 10 batches (@"+str(batch)+"/"+str(nBatches)+") took:"+str(math.floor(time1 - time0))+"s")
                        print("Batch training time: "+str(trainTime)+"s")
                        print("Loss: "+str(losses[len(losses) - 1]))
                        time0 = time1
                        trainTime = 0
                
                        print("Training evaluation")
                        performance = self.model.evaluate(X_train, Y_train)

                    if batch > nBatches:
                        print("Breaking in loop.")
                        break
                    btime1 = time.time()
                    print("Epoch took: "+str(btime1 - btime0)+"s")
                    yield {
                        "state": getKerasModelParameters(self.model.model),
                        "performance": performance,
                        "notes": {"epoch": e, "time": btime1 - btime0}
                    }
                    return
                    pars = getKerasModelParameters(self.model.model)
                    loadKerasModelParameters(self.model.model, pars)
                    
            except KeyboardInterrupt:
                print("Keyboard interrupt detected. Ending trial early.")
                break
