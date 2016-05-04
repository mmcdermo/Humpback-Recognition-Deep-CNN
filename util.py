import csv
import math
import numpy as np

def loadTrainingCSV(filename):
    """
    Convert training CSV into dictionary to lookup labels, keyed off filename
    """
    trainDict = {}
    idDict = {}
    targetSize = -1
    maxWhaleID = 0
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        i = 0 
        for row in reader:
            i += 1
            if i == 1: continue # Skip header row

            # Training data will only include whale ID and fluke/bounding points
            trainDict[row[0]] = map(float, row[1:])

            # Update targetSize, excluding bounding box points. This can
            # be done more tactfully later. 
            targetSize = len(row) - 6

            # Update max whale ID if appropriate
            if int(row[1]) > maxWhaleID:
                maxWhaleID = int(row[1])

            # Assume first section of image name (before hyphen) is real whale ID
            idDict[row[1]] = row[0].split("-")[0]
            
    return (trainDict, idDict, targetSize, maxWhaleID)

#
#  Data processing
# 
def quantizeValue(floatVal, numBins):
    """
    Convert a float value in range [0, 1] to a one-hot encoding
    quantized into `numBins` bins
    """
    binNum = math.floor(float(floatVal) * float(numBins))
    quantized = []
    for i in range(numBins):
        if i == binNum:
            quantized.append(1.)
        else:
            quantized.append(0.)
    return np.array(quantized)

def dequantizeValue(softmaxOutput):
    """
    Convert the softmax prediction of a one-hot target into a 
    float value in range [0, 1]
    """
    maxVal = 0
    maxPos = -1
    for i in range(len(softmaxOutput)):
        if softmaxOutput[i] > maxVal:
            maxVal = softmaxOutput[i]
            maxPos = i
    return (float(maxPos) + 0.5) / float(len(softmaxOutput))


#
#  Image processing
# 

def drawVertLine(image, color, x, w):
    return drawWhen(image, color, lambda p_x, p_y: p_x >= x - w/2 and p_x <= x + w/2)

def drawVertLineRel(image, color, xRatio, w):
    return drawVertLine(image, color, xRatio * len(image[0]), w)

def drawHorizLine(image, color, y, h):
    return drawWhen(image, color, lambda p_x, p_y: p_y >= y - h/2 and p_y <= y + h/2)

def drawHorizLineRel(image, color, yRatio, h):
    return drawHorizLine(image, color, yRatio * len(image), h)

def drawCircleRel(image, color, xRatio, yRatio, r):
    return drawCircle(image, color, xRatio * len(image[0]), yRatio * len(image), r)

def drawCircle(image, color, x, y, r):
    return drawWhen(image, color, lambda p_x, p_y: ((p_x - x) ** 2 + (p_y - y) ** 2) ** 0.5 < r)
    
def drawWhen(image, color, drawTest):
    """
    Replace image pixel values with `color` whenever `drawTest` returns true
    for the given coordinates.
    
    This function is incredibly inefficient, but convenient, and performance
    doesn't matter in this application. 
    """
    h = len(image)
    w = len(image[0])
    for y in range(h):
        for x in range(w):
            if drawTest(x, y):
                image[y][x] = color
    return image
