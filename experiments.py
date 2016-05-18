import json

import experiment as exp
import models
import images
import evaluate
import util

def getMaxWhaleID(data):
    """
    Extract the max whale ID from training data
    """
    return max(map(lambda x: int(data[x][0]), data))

class VGGStack(exp.KerasExperiment):
    experimentName="VGGStack"

    def loadData(self):
        """
        Helper function to load data given self.envParams
        """
        with open(self.envParams["dataTrain"], 'r') as file:
            self.trainingData = json.loads(file.read())
        with open(self.envParams["dataTest"], 'r') as file:
            self.testingData = json.loads(file.read())
            
    def predictImage(self, filename, trialName=None, stateNum=None):
        # Load saved state
        self.loadTrial(trialName, stateNum)
        
        print("Predicting for file: "+str(filename))
        predicted = evaluate.permutationPredictClass(self.model, self.datagen, filename, 20)
        idSoftmaxOut = predicted[4]
        classOut = evaluate.classOutput(idSoftmaxOut)

        idDict = util.extractIDMapping(self.trainingData)
        
        print("Top 10 identity predictions: ")
        for i in range(10):
            print("Whale "+idDict[str(classOut[i][0])]+" (probability "+str(classOut[i][1])+")")
            #pruned = filter(lambda x: str(x) in idDict, classOut[:20])
        return map(lambda x: {"whaleID": idDict[str(x[0])], "probability": x[1]}, classOut[:20])

    def evaluate(self, trialName=None, stateNum=None):
        self.loadTrial(trialName, stateNum)
        predictions = evaluate.evaluatePermutations(self.trainingData, self.testingData, self.model, self.datagen)
        return predictions
                
    def listEnvParams(self):
        return [
            exp.envParam("vggWeights", desc="h5 weights file for complete VGGNet", required=True),
            exp.envParam("imageFolder", desc="Folder containing fluke images", required=True),
            exp.envParam("batchSize", desc="Training batch size (default 16)"),            
            exp.envParam("dataTrain", desc="Training data in JSON format", required=True),
            exp.envParam("dataTest", desc="Testing data in JSON format", required=True),
            exp.envParam("storedWeights", desc="Stored weights to load into the model [temporary arg]"),            
        ]
    
    def genTrialParams(self):
        modelParams = {"denseSize": 256}
        datagenParams = {"denseSize": 256}
        return [{"model": modelParams,
                "datagen": datagenParams
        }]
        
    def setupTrial(self, trialParams):
        self.loadData()
        maxWhaleID = max(getMaxWhaleID(self.trainingData), getMaxWhaleID(self.testingData))
        if "batchSize" not in self.envParams:
            self.envParams["batchSize"] = 16
        modelParams = {"nBins": 20,
                       "maxWhaleId": maxWhaleID,
                       "nCols": 146,
                       "nRows": 256,
                       #"partialStoredWeights": partialStoredWeights,
                       #"partialWeightModelDepth": modelDepth,
                       "model": "vggAdapter",
                       "inputShape": (14, 14, 512),
                       "denseSize": 256,
                       "name": "vggAdapter"
        }
        for k in trialParams["model"]:
            modelParams[k] = trialParams["model"][k]
        print(modelParams)
        self.model = models.BasicModel(modelParams)

        if "storedWeights" in self.envParams and self.envParams["storedWeights"] != None:
            print("--- Trial: Loading weights from file "+self.envParams["storedWeights"])
            self.model.loadWeights(self.envParams["storedWeights"])
        
        datagenParams = {
            "imageFolder": self.envParams["imageFolder"],
            "vggWeights": self.envParams["vggWeights"],
            "dataset": self.trainingData,
            "batchSize": self.envParams["batchSize"],
            "shift": 0.1,
            "rotation": 4,
            "maxScale": 1.00,
            "minScale": 0.95,
            "parallel": False,
            "precomputedData": False,
            "precomputedDataFolder": "",
            "preview": False
        }
        
        for k in trialParams["datagen"]:
            datagenParams[k] = trialParams["datagen"][k]

        self.datagen = images.VGGDatagen(datagenParams)

