"""
 Library for handling zooniverse data (tail fluke points, bounding boxes, etc) for 
 humpback whales
"""
import csv
import json
import numpy as np
import sklearn.cluster
import skimage.transform
import skimage.data
import skimage.io

def islist(x):
    return isinstance(x, list)

def cleanFilename(x):
    """
    Helper function to sanitize a filename for comparison
    """
    return x.lower().replace(" ", "_")

def loadZooniverseData(zooniverseFile):
    """
    Collect zooniverse fluke position data, keyed by filename
    """
    aggregatedData = {
        "flukeTipPoints": {},
        "flukeNotchPoints": {},
        "flukeBoundingBoxes": {}
    }
    
    # Mapping of kaggle task types to our named tasks
    taskMapping = {
        "T1" : "flukeBoundingBoxes",
        "T2": "flukeTipPoints",
        "T3": "flukeNotchPoints"
    }
    
    imageDims = {}
    
    with open(zooniverseFile, 'rbU') as csvfile:
        csvreader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        i = 0
        for row in csvreader:
            anns = json.loads(row['annotations'])
            subj = json.loads(row['subject_data'])
            meta = json.loads(row['metadata'])
            filename = ""
            for k in subj:
                # subject_data has one key that's the subject ID
                if "filename" in subj[k]:
                    filename = subj[k]['filename']
                elif "Filename" in subj[k]:
                    filename = subj[k]['Filename']
                else:
                    print("No filename found")
                    print(subj)
            filename = cleanFilename(filename)
            for ann in anns:
                if ann['task'] in taskMapping and ann['value'] is not None:
                    i = i + 1
                    taskType = taskMapping[ann['task']] # T1, T2, T3
                    if filename not in aggregatedData[taskType]:
                        aggregatedData[taskType][filename] = []
                    if islist(ann['value']):
                        for x in ann['value']:
                            aggregatedData[taskType][filename].append(x)
                    else:
                        aggregatedData[taskType][filename].append(ann['value'])
            if i % 10000 == 0:
                print(".")

    print("Loaded "+str(i)+" zooniverse tasks")
    return aggregatedData

def groupDataIntoClusters(data, clusters):
    """
    Group data into clusters based on indices given by sklearn.cluster.dbscan
    """
    clusteredPoints = []
    for pointIndex in range(len(clusters[1])):
        cluster = clusters[1][pointIndex]
        if cluster == -1: #Noisy sample
            continue
        while cluster >= len(clusteredPoints):
            clusteredPoints.append([])
        clusteredPoints[cluster].append(data[pointIndex])
    return clusteredPoints

def handedPoints(points):
    """
    Determine which point is the left point and which is the right
    """
    if points[0][0] < points[1][0]:
        return {"leftTipPoint": points[0],
                "rightTipPoint": points[1]}
    else:
        return {"leftTipPoint": points[1],
                "rightTipPoint": points[0]}

def removeConcentricBoxes(boxes):
    """
    Takes a list of bounding boxes and removes those that are entirely
    contained inside others.
    """
    isolatedBoxes = []
    for innerBox in boxes:
        contained = False
        for outerBox in boxes:
            if(innerBox[0] > outerBox[0] and #x1_i > x1_o
               innerBox[0] + innerBox[2] < outerBox[0] + outerBox[2] and #x2_i < x2_o
               innerBox[1] > outerBox[1] and #y1_i > y1_o
               innerBox[1] + innerBox[3] < outerBox[1] + outerBox[3]): #y2_i < y2_o):
                contained = True
                print("\tFound concentric bounding box. Removing from consideration.")
                break
        if(not contained):
            isolatedBoxes.append(innerBox)
    return isolatedBoxes


def flukePositionsFromBoundingBoxes(tipGroupedPoints, tipBoundingBoxes, scale):
    """
    Use all tip points and bounding boxes from an image to extract fluke positions
    Returns: [{leftTipPoint: , rightTipPoint: , boundingX: , boundingY: , boundingW:, boundingH: }]
    """

    boxPoints = []
    for box in tipBoundingBoxes:
        boxPoints.append([])
    margin = 20 * scale

    # Average and scale the tip grouped points
    avgPoints = map(lambda x: np.mean(x, axis=0) * scale, tipGroupedPoints)

    # Average and scale the bounding box x,y,w,h
    avgBoxes = map(lambda x: np.mean(x, axis=0) * scale, tipBoundingBoxes)
    avgBoxes = removeConcentricBoxes(avgBoxes)

    positions = []

    # Determine which bounding box each point belongs to
    nomatch = []
    for point in avgPoints:
        found = False
        for boxIdx in range(len(avgBoxes)):
            box = avgBoxes[boxIdx]
            if point[0] > box[0] - margin and \
               point[1] > box[1] - margin and \
               point[0] < box[0] + box[2] + margin and \
               point[1] < box[1] + box[3] + margin:
                boxPoints[boxIdx].append(point)
                found = True
        if not found:
            print("\tCould not find a bounding box that encloses point.")
            nomatch.append(point)

    if len(nomatch) > 0:
        #Use flukePositionsFromPoints for points without bounding boxes
        pointPositions = flukePositionsFromPoints(map(lambda x: [x], nomatch), 1)
        positions = positions + pointPositions

    for idx in range(len(boxPoints)):
        boxp = boxPoints[idx]
        if len(boxp) < 2:
            if len(boxp) == 1:
                print("\tFound cut-off fluke at edge of image. Cropping to edge of image.")
                # Use bounding box to recreate missing point
                base = {}
                base["boundingX"] = avgBoxes[idx][0]
                base["boundingY"] = avgBoxes[idx][1]
                base["boundingW"] = avgBoxes[idx][2]
                base["boundingH"] = avgBoxes[idx][3]
                base["leftTipPoint"] = boxp[0]
                base["rightTipPoint"] = [base["leftTipPoint"][0] + base["boundingW"], base["leftTipPoint"][1]]
                positions.append(base)
            else:
                print("\tFound "+str(len(boxp))+" tip points within bounding box. Should have found 1 or 2.")

        else:
            base = handedPoints(boxp)
            base["boundingX"] = avgBoxes[idx][0]
            base["boundingY"] = avgBoxes[idx][1]
            base["boundingW"] = avgBoxes[idx][2]
            base["boundingH"] = avgBoxes[idx][3]
            positions.append(base)
    return positions

def flukePositionsFromPoints(tipGroupedPoints, scale):
    """
    Use all tip points from an image to extract fluke positions
    Returns: [{leftTipPoint: , rightTipPoint: }]
    """

    # Use K-Means to group points together for different flukes
    k = len(tipGroupedPoints)/2

    kFlukeClusters = sklearn.cluster.KMeans(k)
    try:
        kFlukeClusters.fit(map(lambda x: np.mean(x, axis=0), tipGroupedPoints))
    except:
        print("k-means failed to find the "+str(k)+" groupings for image")
        return []

    # For each fluke, extract the two associated positions
    positions = []
    for i in range(len(kFlukeClusters.labels_)/2):
        points = []
        for idx in range(len(kFlukeClusters.labels_)):
            if kFlukeClusters.labels_[idx] == i:
                points.append(tipGroupedPoints[idx])

        if len(points) < 2:
            print("\tFluke group "+str(i)+" has only one point")
            # TODO: Assume tail extends to edge of image and crop from there.
            continue

        # Take the mean of our grouped points and scale them
        points = map(lambda x: np.mean(x, axis=0) * scale, points)

        positions.append(handedPoints(points))
    return positions

# Functions to extract data from kaggle values
dataExtractors = {
    "flukeTipPoints": lambda x: [x["x"], x["y"]],
    "flukeNotchPoints": lambda x: [x["x"], x["y"]],
    "flukeBoundingBoxes": lambda x: [x["x"], x["y"], x["width"], x["height"]]
}

def flukePositions(imagePath, imageData):
    """
    Extract fluke position information given zooniverse tip point
    and bounding box tasks. 
    
    `imageData` should be of the form:
    { flukeTipPoints: [...],
      flukeBoundingBoxes: [...],
      flukeNotchPoints: [...], 
    }

    Returns a list of identified flukes, with aggregated tip points
    notchPoint, and possibly bounding box information
    [{leftTipPoint: , rightTipPoint,  notchPoint: , boundingX:,  ...}, ...]
    """
    points = {}
    clusters = {}

    # Initialize our point arrays for each task type
    for taskType in imageData:
        points[taskType] = []

    # Extract point arrays from kaggle values
    for taskType in imageData:
        for userPoint in imageData[taskType]:
            points[taskType].append(dataExtractors[taskType](userPoint))
            
    # Cluster points together using dbscan
    for taskType in points:
        clusters[taskType] = []
        if len(points[taskType]) == 0:
            continue
        try:
            clusters[taskType] = groupDataIntoClusters(points[taskType], sklearn.cluster.dbscan(points[taskType], eps=20))
        except:
            print("Image "+imagePath+" failed to dbscan for task type "+taskType)
            continue

    # Read the image and calculate our scaling factor
    image = skimage.io.imread(imagePath)
    scale = float(len(image[0])) / 960

    flukePos = []
    if len(clusters['flukeBoundingBoxes']) > 0:
        flukePos = flukePositionsFromBoundingBoxes(clusters['flukeTipPoints'], clusters['flukeBoundingBoxes'], scale)
    elif len(clusters['flukeTipPoints']) > 0:
        flukePos = flukePositionsFromPoints(clusters['flukeTipPoints'], scale)

    # TODO: Add notch points to fluke pos
    for elem in range(len(flukePos)):
        flukePos[elem]['imageWidth'] = float(len(image[0]))
        flukePos[elem]['imageHeight'] = float(len(image))

    return flukePos

def augmentBoundingBoxFromTipPoints(imageData):
    """
    Given {leftTipPoint: [x,y], rightTipPoint: [x,y]}, augment the dictionary
    with a fluke bounding box (boundingX, boundingY, boundingW, boundingH)
    """
    margin = 10
    w = imageData['rightTipPoint'][0] - imageData['leftTipPoint'][0]
    final_w = w + 2 * margin
    final_h = 4.0 / 7.0 * w + 2 * margin

    top = min(imageData['rightTipPoint'][1], imageData['leftTipPoint'][1])

    r = {}
    for k in imageData:
        r[k] = imageData[k]
    r["boundingX"] = max(0, imageData['leftTipPoint'][0] - margin)
    r["boundingY"] = max(0, top - margin)
    r["boundingW"] = final_w
    r["boundingH"] = final_h
    
    return r

def pointsFromBoundingBox(imageData):
    """
    Convert {boundingX: ..., boundingW: ..., ...} to [[x, y], [x, y]]
    """
    return [[imageData['boundingX'], imageData['boundingY']],
            [imageData['boundingX'] + imageData['boundingW'],
             imageData['boundingY'] + imageData['boundingH']]]
