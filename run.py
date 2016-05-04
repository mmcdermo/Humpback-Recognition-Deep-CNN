from __future__ import print_function
from random import randrange

import skimage.io
import skimage.util
import skimage.color
import skimage.transform
import skimage

import time
import sys
import csv
import time
import math
import numpy as np
import argparse

import h5py
from keras.datasets import cifar10
from keras.utils import np_utils

import settings
import util
from models import *
from images import *
from evaluate import *

def main():
    parser = argparse.ArgumentParser(description='Humpback recognition model.')

    # Basic arguments (images, data, stored weights)
    parser.add_argument('--imageDir', dest='imageDir', help="Directory containing training images", required=True)
    parser.add_argument('--trainCSV', dest='trainCSV', help="CSV containing training data", required=True)
    parser.add_argument('--storedWeights', dest='storedWeights', help="h5 file containing keras exported model")

    # Mode arguments (train/evaluate/predict)
    parser.add_argument('--train', dest='train', action='store_const',
                        const=True, default=False,
                        help='Train the model')
    
    parser.add_argument('--evaluate', dest='evaluate', action='store_const',
                        const=True, default=False,
                        help="Evaluate the model's performance. Use with --storedWeights")

    parser.add_argument('--predict', dest='predict', default=False,
                        help="Perform prediction on the given image. Use with --storedWeights")

    # Advanced arguments (for training)
    parser.add_argument('--pretrainedWeights', dest='pretrainedWeights', help="Weights to initialize convolutional layers of the model. [Layerwise training experiments]")
    parser.add_argument('--modelDepth', dest='modelDepth', type=int, help="Maximum depth (in chunks of layers) of model to train. [Layerwise training experiments]") 

    args = parser.parse_args(sys.argv[1:])
    
    imageDir = args.imageDir
    trainCSV = args.trainCSV
    storedWeights = args.storedWeights
    pretrainedWeights = args.pretrainedWeights
    modelDepth = args.modelDepth
        
    trainDict, idDict, targetSize, maxWhaleID = util.loadTrainingCSV(trainCSV)

    # Determine distribution of # images / whale
    distribution = {}
    numPics = {}
    for x in trainDict:
        val = trainDict[x]
        whaleID = val[0]
        if whaleID not in numPics: numPics[whaleID] = 0
        numPics[whaleID] += 1
    for x in numPics:
        n = numPics[x]
        if n not in distribution: distribution[n] = 0
        distribution[n] += 1

    print("Distribution of # pictures / whale: ")
    print(distribution)

    print("Average images per whale: "+str(float(len(trainDict)) / float(maxWhaleID)))

    print("Number individual whales: "+str(maxWhaleID))
    
    partialStoredWeights = None
    if pretrainedWeights != None and modelDepth != None:
        print("+Partially loading stored weights from "+pretrainedWeights+" to a depth of "+str(modelDepth))
        partialStoredWeights = h5py.File(pretrainedWeights, "a")

    model = BasicModel({"nBins": 20,
                        "maxWhaleId": maxWhaleID,
                        "nCols": 146,
                        "nRows": 256,
                        "partialStoredWeights": partialStoredWeights,
                        "partialWeightModelDepth": modelDepth,
                        "model": "vggAdapter",
                        "inputShape": (14, 14, 512),
                        "denseSize": 256,
                        "name": "vggAdapter"
    })
        
    if partialStoredWeights != None:
        partialStoredWeights.close()
    
    if storedWeights != None:
        model.loadWeights(storedWeights)
        print("Fully loaded stored weights from "+storedWeights)

    # Split training data and testing data
    trainingData, testingData = persistTrainingData(trainDict, "humpbackClassification", divideTrainingData)

    trainingData = dict(trainingData)
    testingData = dict(testingData)
    print("Divided into training data and testing data.")
    print("Training data size: "+str(len(trainingData)))
    print("Testing data size: "+str(len(testingData)))

    # Collect our testing data
    X_test = np.array([])
    Y_test = np.array([])

    testDatagen = VGGDatagen({
        "imageFolder": imageDir,
        "dataset": testingData,
        "batchSize": settings.batchSize,
        "shift": 0.1,
        "rotation": 4,
        "maxScale": 1.00,
        "minScale": 0.95,
        "parallel": False,
        "precomputedData": False,
        "precomputedDataFolder": "vggtestoutput",
        "preview": False
    })
            
    trainDatagen = VGGDatagen({
        "imageFolder": imageDir,
        "dataset": trainingData,
        "batchSize": settings.batchSize,
        "shift": 0.05,
        "rotation": 6,
        "maxScale": 1.00,
        "minScale": 0.85,
        "parallel": False,
        "precomputedData": False,
        "precomputedDataFolder": "vggtrainoutput",
        "preview": False
    })

    # Evaluate model performance
    if args.evaluate == True:
        print("Evaluating model performance...")
        predictions = evaluatePermutations(trainingData, trainDict, testingData, model, trainDatagen)
        exit(1)
    
    # Predict fluke positions and image identity
    if args.predict != False:
        filename = args.predict
        print("Predicting whale identity for file: "+str(filename))
        predicted = permutationPredictClass(model, trainDatagen, filename, 20)
        idSoftmaxOut = predicted[4]
        classOut = classOutput(idSoftmaxOut)
        print("Top 10 identity predictions: ")
        for i in range(10):
            print("Whale "+idDict[str(classOut[i][0])]+" (probability "+str(classOut[i][1])+")")

    # Train the model
    if args.train == False:
        exit(1)
    for e in range(settings.nEpoch):
        btime0 = time.time()
        print("=============================== Epoch "+str(e))

        # Evaluate classification performance
        print("Epoch pre-testing evaluation")
        for X_test, Y_test, filenames in testDatagen.generate(1, dataset=testingData):
            model.evaluate(X_test, Y_test)

        batch = 0
        trainTime = 0
        time0 = time.time()
        losses = []

        nBatches = len(trainDict) / settings.batchSize
        for X_train, Y_train, filenames in trainDatagen.generate(nBatches, dataset=trainingData):
            batch += 1
            trainTime0 = time.time()
            loss = model.trainOnBatch(X_train, Y_train)
            trainTime += time.time() - trainTime0
            losses.append(loss)
            
            time1 = time.time()
            if batch % 10 == 0:
                print("Last 10 batches (@"+str(batch)+"/"+str(nBatches)+") took:"+str(math.floor(time1 - time0))+"s")
                print("Batch training time: "+str(trainTime)+"s")
                print("Loss: "+str(losses[len(losses) - 1]))
                model.saveWeights(str(e))
                time0 = time1
                trainTime = 0
                
                print("Training evaluation")
                model.evaluate(X_train, Y_train)
            if batch > nBatches:
                print("Breaking in loop.")
                break

        btime1 = time.time()
        print("Epoch took: "+str(btime1 - btime0)+"s")

if __name__ == "__main__":
    main()
