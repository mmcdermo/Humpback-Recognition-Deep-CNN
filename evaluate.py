"""
 Functions to evaluate model accuracy  
"""

import sys
import skimage.io
import skimage.color
import numpy as np

import util 
import settings

from images import humpbackImages, reshapeKeras, preprocessImage, augmentImage
from util import dequantizeValue

def classOutput(softmaxOutput):
    """
    Returns a list of tuples of IDs and their probability, sorted by probability
    """
    return sorted([ (i, softmaxOutput[i]) for i in range(len(softmaxOutput))], key=lambda x: x[1], reverse=True)

    
def permutationPredictClasses(model, datagen, trainDict, imageNames, numPermutations=5):
    """
    Sum softmax predictions for all given training instances
    """
    print("Generating "+str(numPermutations)+" permutations per image ("+str(len(imageNames))+" images)...")
    inputBatch = []
    i = 0
    for imageName in imageNames:
        if i % 50 == 0:
            print(".")
        i += 1
        for n in range(numPermutations):
            x, y, files = datagen.generateBatch({ imageName: trainDict[imageName] })
            inputBatch.append(x[0])

    print("Predicting classes from permutations...")
    print(np.array(inputBatch).shape)
    outputs = model.predict(np.array(inputBatch))

    print("Averaging over predictions... ")
    perImagePredictions = []
    for i in range(len(imageNames)):
        startIdx = int(i * numPermutations)
        endIdx = int((i + 1) * numPermutations)

        imSlice = []
        for j in range(len(outputs)):
            imSlice.append([])
            for n in range(numPermutations):
                imSlice[j] = outputs[j][startIdx:endIdx]

        averaged = []
        for output in imSlice:
            averaged.append(np.average(output, axis=0))
        perImagePredictions.append(averaged)
            
    return perImagePredictions
    
def permutationPredictClass(model, datagen, filename, numPermutations=20):
    """
    Generate `numPermutations` permutations of the given image and perform
    prediction with the model, averaging over the softmax outputs 
    """

    try:
        image = skimage.io.imread(filename, "pillow")
    except:
        print("Failed to load image "+filename)
        exit(1)

    inputBatch = []
    for n in range(numPermutations):
        x, y, files = datagen.generateBatch({ filename: [0, 0.1, 0.1, 0.8, 0.1] },
                                            imageFolder="./")
        inputBatch.append(x[0])

    summed = []
    outputs = model.predict(np.array(inputBatch))
    for output in outputs:
        summed.append(np.average(output, axis=0))
    return summed
        
def evaluateTopN(model, datagen, trainDict, imageDir, imageFiles):
    """
    Get the positions of the correct label in sorted softmax output
    for the given batch of images. 
    """
    positions = []
    for image in imageFiles:
        # Preprocess image
        preprocessed = preprocess(imageDir + "/" + image, datagen)
    
        # Perform prediction 
        prediction = model.predict({"image_input": np.array([preprocessed])})
        regression = prediction['regression_output'][0]
        whaleID = prediction['id_output'][0]
        target = map(lambda x: float(x), trainDict[image][2:])

        classes = classOutput(whaleID)
        realClass = trainDict[image][0]
        pos = -1
        for i in range(len(classes)):
            if int(classes[i][0]) == int(realClass):
                pos = i
        if pos == -1:
            print("Error with softmax whale ID for file "+imageFile)
        else:
            positions.append(pos)
    return positions
        
def preprocess(imageFile, datagen):
   """
   Preprocess an image in preparation for evaluation
   """
   image = skimage.io.imread(imageFile)
   preprocessed = reshapeKeras(preprocessImage(image, settings.imgRows, settings.imgCols))
   datagenned = None
   for X, Y in datagen.flow(np.array([preprocessed]), np.array([0]), batch_size=1):
       datagenned = X
       break
   return datagenned[0]

def visualPredictions(image, predictedValues, color=1., radius=5):
    visual = util.drawCircleRel(image, color, predictedValues['flukeLeftX'],
                                predictedValues['flukeLeftY'], radius)
    visual = util.drawCircleRel(image, color, predictedValues['flukeRightX'],
                                predictedValues['flukeRightY'], radius)
    return visual

def stackImages(image1, image2, padding=1):
    """
    Stack two images together, with padding zero-d pixels separating them
    """
    paddedBar = np.zeros((padding, image1.shape[1]))
    return np.concatenate([image1, paddedBar, image2], axis=0)

def exampleImage(trainDict, classID):
    """
    Returns the filename of an example image of a class with the given ID
    """
    for x in trainDict:
        val = trainDict[x]
        whaleID = val[0]
        if int(whaleID) == int(classID):
            return x

def drawUnlabeledPrediction(trainDict, imageFolder, filename, prediction):
    """
    Given the output of an image from `evaluatePermutations`, returns an image
    representation of tip point and identity predictions. 
    """
    targetImagefile = exampleImage(trainDict, prediction['targetClass'])
    predictedImagefile = exampleImage(trainDict, prediction['predictedClass'])

    trainingImage = preprocessImage(skimage.io.imread(imageFolder + "/" + filename, "pillow"), settings.imgRows, settings.imgCols)
    targetImage = preprocessImage(skimage.io.imread(imageFolder + "/" + targetImagefile, "pillow"), settings.imgRows, settings.imgCols)
    predictedImage = preprocessImage(skimage.io.imread(imageFolder + "/" + predictedImagefile, "pillow"), settings.imgRows, settings.imgCols)
    
    # Overlay tip point targets onto trainingImage
    target = map(float, prediction["targetData"])
    trainingImage = visualPredictions(trainingImage, {
        "flukeLeftX": target[1],
        "flukeLeftY": target[2],
        "flukeRightX": target[3],
        "flukeRightY": target[4]
    }, color=1., radius=10)    
    
    # Overlay tip point predictions onto trainingImage
    predicted = prediction["softmaxOutputs"]

    trainingImage = visualPredictions(trainingImage, {
        "flukeLeftX": dequantizeValue(predicted[0][0]),
        "flukeLeftY": dequantizeValue(predicted[1][0]),
        "flukeRightX": dequantizeValue(predicted[2][0]),
        "flukeRightY": dequantizeValue(predicted[3][0])
    }, color=0., radius=6)

    finalImg = stackImages(trainingImage, targetImage)
    finalImg = stackImages(finalImg, predictedImage)

    return finalImg

def histogram(arr, bins):
    hMin = min(arr)
    hMax = max(arr)
    step = float(hMax - hMin) / float(bins)
    hist = []
    for i in range(bins+1):
        hist.append(0)
    for el in arr:
        hBin = int((el - hMin) / step)
        hist[hBin] += 1
    return (hist, hMin, hMax, step)
    
def evaluatePermutations(trainingData, trainDict, testingData, model, trainDatagen):
        numPics = {}
        maxWhaleID = -1
        for x in trainingData:
            val = trainingData[x]
            whaleID = val[0]
            if whaleID not in numPics: numPics[whaleID] = 0
            numPics[whaleID] += 1
            if whaleID > maxWhaleID: maxWhaleID = whaleID            

        numPicsPos = []
        positions = []
        predictions = {}
        manyPicsPos = []
        z = 0

        numTest = 400
        names = testingData.keys()[0:numTest]
        allSoftmaxOutputs = permutationPredictClasses(model, trainDatagen, trainDict, names, 1)
        # Track the softmax probabilities for matches and match failures
        matchProbabilities = []
        noMatchProbabilities = []
        for imageName in testingData:
            z += 1
            if z > numTest: break
            softmaxOutputs = allSoftmaxOutputs[z-1]
            idSoftmaxOut = softmaxOutputs[4]
            classOut = classOutput(idSoftmaxOut)
            targetClass = testingData[imageName][0]
            predictions[imageName] = {"predictedClass": classOut[0][0], "targetClass": targetClass, "targetData": testingData[imageName], "softmaxOutputs": softmaxOutputs}
            pos = -1
            for i in range(len(classOut)):
                if int(classOut[i][0]) == int(targetClass):
                    pos = i
            if pos == -1:
                print("Error with softmax ID for file "+imageName)
                print(idSoftmaxOut.shape)
                print(classOut)
                print(targetClass)
                exit(1)
            else:
                if pos == 0:
                    matchProbabilities.append(classOut[0][1])
                else:
                    noMatchProbabilities.append(classOut[0][1])
                
                print("Pos: "+str(pos))
                print("Npix:"+str(numPics[targetClass]))
                positions.append(pos)
                numPicsPos.append([pos, numPics[targetClass]])
                if numPics[targetClass] > 3:
                    manyPicsPos.append(pos)
                    
        avgPos = float(sum(positions)) / float(len(positions))
        print("Average correct class position: "+str(avgPos)+"/"+str(maxWhaleID))
        print("Pct top 10: "+str(1. * len(filter(lambda x: x < 10, positions)) / float(len(positions))))
        print("Pct top 5: "+str(1. * len(filter(lambda x: x < 5, positions)) / float(len(positions))))
        print("Pct top 3: "+str(1. * len(filter(lambda x: x < 3, positions)) / float(len(positions))))
        print("Pct top 1: "+str(1. * len(filter(lambda x: x < 1, positions)) / float(len(positions))))
        
        avgPos = float(sum(manyPicsPos)) / float(len(manyPicsPos))
        print("Many Pics Pos Average correct class position: "+str(avgPos)+"/"+str(maxWhaleID))
        print("Pct top 10: "+str(1. * len(filter(lambda x: x < 10, manyPicsPos)) / float(len(manyPicsPos))))
        print("Pct top 5: "+str(1. * len(filter(lambda x: x < 5, manyPicsPos)) / float(len(manyPicsPos))))
        print("Pct top 3: "+str(1. * len(filter(lambda x: x < 3, manyPicsPos)) / float(len(manyPicsPos))))
        print("Pct top 1: "+str(1. * len(filter(lambda x: x < 1, manyPicsPos)) / float(len(manyPicsPos))))

        print("Softmax output distribution for matches")
        print(histogram(matchProbabilities, 10))

        print("Softmax output distribution for failures")
        print(histogram(noMatchProbabilities, 10))

        return predictions

indices=['flukeLeftX','flukeLeftY','flukeRightX','flukeRightY','bounding1X','bounding1Y','bounding2X','bounding2Y']
