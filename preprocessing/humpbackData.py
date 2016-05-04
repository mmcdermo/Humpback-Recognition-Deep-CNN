"""
humpbackData.py

Uses the given zooniverse data and images to generate training data
for our keras model

Usage: python humpbackData.py <zooniverse file> <image directory> <max training size | default=10000>
"""

import csv
import json
import re
import os
import sys
from math import cos, sin, degrees
from subprocess import call
import shutil
import os, time
import datetime
import numpy as np
import sklearn.cluster
import skimage.transform
import skimage.data
import skimage.io

from humpbackZooniverse import *

def loadImageInfo(flukeImagesDir):
    """
    Acquire image information from given directory
    
    Whale IDs are acquired either from the directory structure, or the image name. 

    We assume that any images in the image directory have their ID encoded in the name,
    as the first substring before a hyphen - e.g. 11622 in "11622-20140602-JAC-0044.JPG"

    Otherwise, subfolder names are assumed to be the whale IDs (11622 in 11622/foo.jpg)
    """
    images = []
    for dirname, dirnames, filenames in os.walk(flukeImagesDir):
        path = filter(lambda x: len(x) > 0, dirname.split('/'))
        for filename in filenames:
            if filename[0] == ".":
                #Ignore hidden files
                continue
            whaleID = path[len(path) - 1]
            parts = filename.split("-")
            # If files are directly inside given directory, get whale ID from name
            if len(path) < 2:
                whaleID = parts[0]
            images.append({
                "whaleID": whaleID,
                "filename": cleanFilename(filename),
                "strippedFilename": cleanFilename("-".join(parts[1:len(parts)])),
                "path": ("/").join(path) + "/" + filename,
            })
    return images

def usage():
    print("Usage: python humpbackData.py <zooniverse file> <fluke image directory>")
    exit(1)

def IDMapping(dataset):
    """
    Create a mapping of whale IDs to UIDs starting at 0
    """
    IDMapping = {}
    i = 0
    for data in dataset:
        if not data['image']['whaleID'] in IDMapping:
            IDMapping[data['image']['whaleID']] = i
            i += 1
    print("Found "+str(i)+" unique whale IDs")
    return IDMapping
    
def copyTrainingImages(imageInfo):
    """
    Creates directory containing images needed for training
    """
    print("Copying images...")
    i = 0
    try:
        os.mkdir("trainingImages")
    except:
        print("trainingImages directory already exists. Continuing.")
    for image in imageInfo:
        i += 1
        if i % 200 == 0:
            print(".")
        shutil.copyfile(image['path'], 'trainingImages/'+image['filename'])

def createCSV(dataset, mapping):
    """
    Create trainingData.csv in the format needed by our keras implementation
    """
    with open('trainingData.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['name', 'whaleID', 'flukeLeftX', 'flukeLeftY', 'flukeRightX', 'flukeRightY', 'bounding1X', 'bounding1Y', 'bounding2X', 'bounding2Y'])
        for data in dataset:
            imgWidth = data['flukeData']['imageWidth']
            imgHeight = data['flukeData']['imageHeight']
            boundingPoints = pointsFromBoundingBox(data['flukeData'])
            writer.writerow([data['image']['filename'], mapping[data['image']['whaleID']],
                             data['flukeData']['leftTipPoint'][0] / imgWidth,
                             data['flukeData']['leftTipPoint'][1] / imgHeight,
                             data['flukeData']['rightTipPoint'][0] / imgWidth,
                             data['flukeData']['rightTipPoint'][1] / imgHeight,
                             boundingPoints[0][0] / imgWidth,
                             boundingPoints[0][1] / imgHeight,
                             boundingPoints[1][0] / imgWidth,
                             boundingPoints[1][1] / imgHeight,
            ])

def main():
    if len(sys.argv) < 3:
        usage()
    zooniverseFile = sys.argv[1]
    flukeImagesDir = sys.argv[2]
    maxTrainingSize = -1
    try:
        maxTrainingSize = int(sys.argv[3])
        print("Max training size set to "+str(maxTrainingSize))
    except:
        pass
    imageInfo = loadImageInfo(flukeImagesDir)
    aggregatedData = loadZooniverseData(zooniverseFile)

    dataset = []
    i = 0
    for image in imageInfo:
        i += 1
        flukesData = []

        #TODO: Offer choice of random sampling rather than first N samples
        if maxTrainingSize != -1 and i > maxTrainingSize:
            break

        # Determine which filename to use (different versions of the same filename
        #  often appear in the zooniverse data).
        # Here, we base the decision on flukeTipPoint data, since it's almost always
        # available and is essential to the training process
        filename = image['filename']
        if filename not in aggregatedData['flukeTipPoints']:
            if image['strippedFilename'] in aggregatedData['flukeTipPoints']:
                filename = image['strippedFilename']
            else:
                continue

        tipPointData = aggregatedData['flukeTipPoints'][filename]
        boundingBoxData = []
        notchPointData = []
        if filename in aggregatedData['flukeBoundingBoxes']:
            boundingBoxData = aggregatedData['flukeBoundingBoxes'][filename]
        if filename in aggregatedData['flukeNotchPoints']:
            notchPointData = aggregatedData['flukeNotchPoints'][filename]

        flukes = flukePositions(image['path'], {
            "flukeTipPoints": tipPointData,
            "flukeBoundingBoxes": boundingBoxData,
            "flukeNotchPoints": notchPointData
        })

        if len(flukes) != 1:
            print("Skipping image "+image['filename']+". Found "+str(len(flukes))+" flukes (!= 1)")
            continue

        # Augment fluke data with extrapolated bounding box info if necessary
        fluke = flukes[0]
        if "boundingX" not in flukes[0]:
            fluke = augmentBoundingBoxFromTipPoints(fluke)

        dataset.append({
            "image": image,
            "flukeData": fluke
        })

    print("Found "+str(len(imageInfo))+" fluke images.")
    print("A total of "+str(len(dataset))+" fluke images present in zooniverse data.")

    # Generate mapping from whale IDs to integers
    mapping = IDMapping(dataset)
    
    # Create  CSV
    createCSV(dataset, mapping)

    # Copy images for training
    #copyTrainingImages(imageInfo)
    
if __name__ == "__main__":
    main()
