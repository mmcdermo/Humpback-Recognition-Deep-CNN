"""
  Utilities to generate datasets from humpback images
"""
import os
import os.path
import random
import math
from math import cos, sin, degrees, radians
import json
import Queue as Q
from multiprocessing import Queue, Process
import threading
import time

import scipy.ndimage as ndimage
import h5py
import skimage.io
import skimage.util
import skimage.color
import skimage.transform
import skimage
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

import settings
import models

def persistTrainingData(trainDict, basename, generator):
    """
    Returns persisted training and testing data if stored.
    If not, it generates new data with generator(trainDict) 
    and stores it.
    """
    trainingData = {}
    testingData = {}
    
    try:
        with open(basename+'_trainingData.json', 'r') as file:
            raw=file.read()
            trainingData = json.loads(raw)
        with open(basename+'_testingData.json', 'r') as file:
            raw=file.read()
            testingData = json.loads(raw)
    except IOError, ValueError:
        print("Generating new training data...")
        trainingData, testingData = generator(trainDict)
        with open(basename+'_trainingData.json', 'w') as file:
            json.dump(trainingData, file)
        with open(basename+'_testingData.json', 'w') as file:
            json.dump(testingData, file)
    return trainingData, testingData
    
def divideTrainingData(trainDict):
    """
    Divides a dictionary [imgName -> data] into two: training data,
    and testing data. The testing data is only chosen from images of
    whales that have > 1 image in the data set. 

    Returns (trainingData, testingData)
    """

    trainingData = {}
    testingData = {}
    
    # Assemble a dictionary of candidate images containing only
    #  images of whales who are represented multiple times in the data set
    numPics = {}
    trainingDataCandidates = {}
    for x in trainDict:
        val = trainDict[x]
        whaleID = val[0]
        if whaleID not in numPics: numPics[whaleID] = 0
        numPics[whaleID] += 1
        if numPics[whaleID] > 1:
            trainingDataCandidates[x] = numPics[whaleID]

    # Randomly choose a sample of training images from the candidates
    keys = trainingDataCandidates.keys()
    sampleSize = int(math.floor( 0.1 * len(keys)))
    trainingKeys = np.random.choice(np.array(keys), size=sampleSize, replace=False)

    # Assemble the data
    for imageName in trainDict:
        if imageName in trainingKeys:
            testingData[imageName] = trainDict[imageName]
        else:
            trainingData[imageName] = trainDict[imageName]    
        
    return (trainingData, testingData)

class ParallelDatagen():
    """
    This base class provides the ability to generate augmented data
    in parallel with training
    """

    queue = None
    params = None
    exitFlag = False
    currentDataset = None
    dataGen = None
    threads = []
    
    def __init__(self, params):
        # The queue will block when its max size is reached.
        # Here we set the maxsize to 4, so we will only process
        # at most 3 data batches ahead
        self.queue = Queue(maxsize=4)
        self.params = params
        self.initialize()

    def initialize(self):
        """ Override this function if necessary"""
        pass

    def generateBatch(self):
        """ 
        - Override this function -
        Your function should return a single batch of training data,
        given self.params
        """
        time.sleep(2)
        return [1, 2, 3]
    
    def runGenData(self):
        while True:
            try:
                self.queue.put(self.generateBatch(self.currentDataset), block=True, timeout=30)
            except Q.Full:
                print("Timed out adding item to queue")

    def generate(self, numBatches, batchSize=None, dataset=None):
        """ 
        Yields data from a thread safe queue, and starts the 
        generative process
        """
        self.currentDataset = dataset
        
        if self.params['parallel'] == False:
            print("Generating data in serial - no parallelism")
            for i in range(numBatches):
                yield self.generateBatch(dataset)
            return

        if self.dataGen == None:
            self.dataGen = Process(target=self.runGenData, args=())
            self.dataGen.start()
        
        i = 0
        while i < numBatches:
            i += 1
            item = None
            try:
                item = self.queue.get(block=True, timeout=30)
                yield item
            except Q.Empty:
                print("Item retrieval timed out.")
                print(self.queue.qsize())

class HumpbackImagegen(ParallelDatagen):
    """
    Generates augmented humpback image training data, and augments the
    positional data to match. 
    
    Shifts and rotations in the base image are reflected in the
    relative [0,1] position data for flukes

    Required parameters: {
      imageFolder: directory containing training images,
      dataset: dictionary of [imageName -> imageData],
      batchSize: number of augmented images to generate in each batch
    }
    
    Optional parameters: {
      rotation: [0, 360] in degrees (default 0),
      shift: maximum pixels to shift (default 0),
      minScale: Smallest value the scaling can take (default 1),
      maxScale: Largest value the scaling can take (default 1),
      reflect: Whether or not to reflect the data with 0.5 probability (default False)
    }
    """

    def generateBatch(self, dataset=None, imageFolder=None):
        if imageFolder == None:
            imageFolder = self.params['imageFolder']
        if dataset == None:
            dataset = self.params['dataset']

        batchSize = self.params['batchSize']            
        rotation = 0
        shift = 0
        minScale = 1.0
        maxScale = 1.0
        reflect = False
        if "rotation" in self.params:
            rotation = self.params['rotation']
        if "shift" in self.params:
            shift = self.params['shift']
        if "minScale" in self.params:
            minScale = self.params['minScale']
        if "maxScale" in self.params:
            maxScale = self.params['maxScale']
        if "reflect" in self.params:
            reflect = self.params['reflect']

        n = 0
        collectedFilenames = []
        collectedData = []
        collectedLabels = []

        filenames = dataset.keys()
        random.shuffle(filenames)
        for filename in filenames:
            n += 1
            image = None
            try:                
                image = skimage.io.imread(imageFolder + "/" + filename, "pillow")
            except:
                print("Failed to load image "+filename)
                continue
            resized = preprocessImage(image, settings.imgRows, settings.imgCols)
            d = dataset[filename]
            points = [[d[1], d[2]], [d[3], d[4]]]
            augmentedImage, newPoints = augmentImage(resized, points, shift, rotation, minScale, maxScale, reflect)
            collectedFilenames.append(filename)
            collectedData.append(augmentedImage)
            if 'preview' in self.params and self.params['preview'] == True:
                skimage.io.imsave(imageFolder+"../preprocessed/"+filename, augmentedImage)
            label = map(float, [d[0], newPoints[0][0], newPoints[0][1], newPoints[1][0], newPoints[1][1]])
            collectedLabels.append(np.array(label))

            if n == batchSize or n >= len(filenames) - 1:
                return np.array(collectedData), np.array(collectedLabels), collectedFilenames

        print("Returning empty batch.")
        return []

class VGGDatagen(ParallelDatagen):
    """
    Generate data using HumbackImagegen, then run data through pretrained VGG.
    If params['precomputedData'] == True, then data is loaded from 
    params['precomputedDataFolder'] instead of generated on the fly.

    Precomputed data can be used for rapid testing as it greatly reduces 
    training time. 
    """
    hbdatagen = None
    def initialize(self):
        if self.params['precomputedData'] == True:
            self.files = []
            for dirname, dirnames, filenames in os.walk(self.params['precomputedDataFolder']):
                for filename in filenames:
                    self.files.append(filename)
        else:
            # Copy our params to use for the HunchbackImagegen
            # However, we'd like that datagen to be parallel. 
            hbParams = {}
            for k in self.params:
                if k != "parallel":
                    hbParams[k] = self.params[k]
                    
            self.hbDatagen = HumpbackImagegen(hbParams)
            self.vggModel = models.VGGModel({'h5weights': 'VGG/FullVGGWeights.h5',
                                             'modelDepth': 4 })

    def generatePrecomputed(self, dataset=None):
        images, labels, filenames = self._generateBatch(dataset)
        for idx in range(len(filenames)):
            filename = filenames[idx]
            image = images[idx]
            h5f = h5py.File(self.params['precomputedDataFolder'] + "/" + self.augmentFilename(filename), "a")
            h5f["data"] = image
            h5f["label"] = labels[idx]
            h5f.close()

    def _generateBatch(self, dataset=None, imageFolder=None):
        images, labels, filenames = self.hbDatagen.generateBatch(dataset, imageFolder)
        
        # Resize the images and convert them to RGB
        resized = []
        for idx in range(len(images)):
            image = images[idx]
            image = preprocessImage(image, 224, 224)
            image = skimage.color.gray2rgb(image)
            image = reshapeKeras(image)
            resized.append(image)

        predictions = self.vggModel.predict(np.array(resized))
        predictions = np.swapaxes(np.array(predictions), 1, 3)
        return predictions, labels, filenames

    def origFilename(self, filename):
        """ Return the original filename from a precomputed, stored version """
        return filename.split("___")[0]

    def augmentFilename(self, filename):
        """ Return a unique filename in which the original is recoverable """
        return filename + "___" + str(random.randint(1, 100000000)) + ".h5"

    def randomImageSubset(self):
        """
        Returns a pseudo-random subset of the saved data.
        The algorithm first chooses a random (batchSize ** 2) * 4 block of files,
        then chooses batchSize random images from within those. 

        This is done to speed disk-reads so that images are more likely to be 
        sequentially stored on disk. Otherwise, access times and disk load
        can be an issue, since individual files may be spread out over half
        the hard drive. 
        
        This approach assumes that generated files are listed in the order of their
        inodes on disk. 
        """
        blockSize = (self.params['batchSize'] ** 2) * 4
        blockIdx = np.random.randint(0, len(self.files)/blockSize)
        indices = np.random.random_integers(low=blockIdx * blockSize,
                                            high=min(len(self.files) - 1, (blockIdx + 1) * blockSize),
                                            size=self.params['batchSize'])
        indices.sort()
        imageFiles = []
        for i in indices:
            imageFiles.append(self.files[i])
        return imageFiles
        
    def generateBatch(self, dataset=None, imageFolder=None):
        if self.params['precomputedData'] == True:
            imageFiles = np.random.choice(np.array(self.files), size=self.params['batchSize'], replace=False)
            imageFiles = self.randomImageSubset()
            data = []
            labels = []
            filenames = []
            for filename in imageFiles:
                h5f = h5py.File(self.params['precomputedDataFolder'] + "/" + filename)
                x = np.zeros((14, 14, 512))
                y = np.zeros(h5f["label"].shape)
                h5f["data"].read_direct(x)
                h5f["label"].read_direct(y)
                h5f.close()
                data.append(x)
                labels.append(y)
                filenames.append(self.origFilename(filename))
            return np.array(data), np.array(labels), filenames
        else:
            return self._generateBatch(dataset, imageFolder)
    

def augmentImage(image, points, shift, rotation, minScale=1.0, maxScale=1.0, reflect=False):
    """
    Augment the image with a combination of shifts and rotations.
    If this causes any points to go out of bounds, we try again.
    """

    points = map(lambda x: map(lambda y: float(y), x), points)
    def testBounds(points):
        for point in points:
            for dim in point:
                if dim < 0.05 or dim > 0.95:
                    return False
        return True

    newPoints = [[-1,-1]]
    chosenScale = 1.0
    doReflect = False
    shiftX = shiftY = angle = 0
    i = 0
    while not testBounds(newPoints):
        i += 1
        if i > 30:
            #print("Required more than 30 augmentation iterations. Returning early")
            return (image, points)

        # Generate our transformation parameters
        angle = np.random.uniform(-rotation, rotation)
        shiftX = np.random.uniform(-shift, shift) * image.shape[1]
        shiftY = np.random.uniform(-shift, shift) * image.shape[0]
        chosenScale = np.random.uniform(minScale, maxScale)
        doReflect = True if reflect and random.choice([True, False]) else False

        # Translate our points according to those parameters
        newPoints = []
        for point in points:
            newPoint = [point[0], point[1]]
            if doReflect:
                newPoint = [1. - point[0], point[1]]
            newPoint = [newPoint[0] * chosenScale, newPoint[1] * chosenScale]
            newPoint = shiftPoint(newPoint, image, shiftX, shiftY)            
            newPoint = rotatePoint(newPoint, radians(angle))


            newPoints.append(newPoint)

    img = np.copy(image)
    if doReflect:
        img = imgReflect(img, 1)
    img = imgScale(img, chosenScale)        
    img = imgShift(img, shiftX, shiftY, rowIndex=0, colIndex=1)
    img = imgRotation(img, angle, axes=(0,1))    

    
    return img, newPoints

def shiftPoint(point, image, shiftX, shiftY):
    """ Shift a relatively positioned point in image by shiftX and shiftY pixels """
    point[0] += float(shiftX) / float(len(image[0]))
    point[1] += float(shiftY) / float(len(image))
    return point
    
def rotatePoint(point, rads):
    """ Rotate a point [ [0, 1], [0, 1]] in an image about its center """
    w = 1.0
    h = 1.0

    # Change coordinate systems to make (0,0) the image center
    p2 = [point[0], point[1]]
    p2[0] = p2[0] - w/2.0
    p2[1] = p2[1] - h/2.0

    # Rotate the point about the center using a rotation matrix
    p2 = np.matrix([[p2[0]], [p2[1]]])
    R = np.matrix([[cos(rads), sin(rads)], [-1 * sin(rads), cos(rads)]])
    rotated = R * p2

    # Recenter the point about the old coordinate system
    return [float(rotated[0]) + w/2.0, float(rotated[1]) + h/2.0]

def imgReflect(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def imgScale(x, scale):
    initialShape = x.shape
    newShape = map(lambda x: int(math.floor(x * scale)), x.shape)
    img = skimage.transform.resize(x, newShape)
    if scale < 1:
       img =  skimage.util.pad(img, ((0,initialShape[0] - newShape[0]),
                                       (0,initialShape[1] - newShape[1])),
                               mode='edge')
    else:
        img = img[:x.shape[0], :x.shape[1]]
    return img


def imgRotation(x, angle, fill_mode='nearest', cval=0., axes=(0, 1)):
    x = ndimage.interpolation.rotate(x, angle,
                                     axes=axes,
                                     reshape=False,
                                     mode=fill_mode,
                                     cval=cval)
    return x

def imgShift(x, shiftX, shiftY, fill_mode='nearest', cval=0., rowIndex=0, colIndex=1):
    x = ndimage.interpolation.shift(x, (shiftY, shiftX),
                                    order=0,
                                    mode=fill_mode,
                                    cval=cval)
    return x
                                                        

def preprocessImage(image, xDim, yDim):
    """
    Converts a given image to xDim x yDim
    Pads as necessary 
    Converts to grayscale if necessary
    """
    w = float(len(image[0]))
    h = float(len(image))
    xDim = float(xDim)
    yDim = float(yDim)

    # Adjust coordinates to yield the desired ratio
    currentRatio = w / h
    desiredRatio = xDim / yDim
    paddingX = 0
    paddingY = 0
    
    if currentRatio < desiredRatio:
        # (a+x)/b = c/d
        # x = bc/d - a
        paddingX = h * xDim / yDim - w
    else:
        # a/(b+x) = c/d
        # x = ad/c - b
        paddingY = w * yDim / xDim - h

    # Convert values to floats
    image = skimage.img_as_float(image)
    
    # Convert grayscale to RGB if necessary M&
    #if image.ndim == 2:
    #    image = skimage.color.gray2rgb(image)

    if image.ndim == 3: #M&
        image = skimage.color.rgb2gray(image)
        
    # Convert to symmetric padding
    paddingX = int(math.floor(paddingX/2.))
    paddingY = int(math.floor(paddingX/2.))
    
    # Pad and resize image
    padded = skimage.util.pad(image, ((paddingY,paddingY), (paddingX,paddingX)), mode='constant') #(0,0) M&
    resized = skimage.transform.resize(padded, (xDim, yDim)) # ,3 M&

    return resized
    
    
def reshapeKeras(img):
    """ Make the first axis the channel axis of the image """
    return np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)

def unReshapeKeras(img):
    return np.swapaxes(np.swapaxes(img, 0, 2), 0, 1)
