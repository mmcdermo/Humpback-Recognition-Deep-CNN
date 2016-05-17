"""
  Models for humpback whale recognition
"""
import math

import numpy as np
import time
import skimage.color

from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.regularizers import l2, activity_l2
from keras.optimizers import SGD, Adadelta, Adagrad, RMSprop

from images import *
from evaluate import classOutput
from util import dequantizeValue, quantizeValue
import vggModel
    
class CustomModel():
    """
    The CustomModel class groups preprocessing, evaluation, and settings in one place. 
    """
    model = {}
    lastEpoch = 0
    
    def __init__(self, params):
        self.initializeModel(params)

    def name(self):
        return "CustomModel"

    def initializeModel(self, params):
        print("Model initialization has not been implemented")
        
    def loadWeights(self, modelFile=None):
        if modelFile is None:
            # Future: Scan directory for most recently saved weights
            print("Automatic loading of recent weights not yet implemented")
            pass
        else:
            self.model.load_weights(modelFile)

    def saveWeights(self, annotation):
        self.model.save_weights("model_"+self.name()+"_"+annotation+".h5", overwrite=True)
            
    def preprocessX(self, X):
        """ Preprocess inputs to the model """
        return X

    def preprocessY(self, Y):
        """ Preprocess model targets """
        return Y
    
    def evaluate(self, X_test):
        return self.model.predict(self.preprocessX(X_test))

    def trainOnBatch(self, X_train, Y_train):
        return self.model.train_on_batch(self.preprocessX(X_train),
                                         self.preprocessY(Y_train))
    def evaluate(self, X_test, Y_test):
        return self.model.evaluate(self.preprocessX(X_test),
                                   self.preprocessY(Y_test))

    def predict(self, X_test):
        return self.model.predict(self.preprocessX(X_test))

class BasicModel(CustomModel):
    """
    This is a flexible model that serves as a testbed for experiments.
    Given a base model topology to build off of (bigModel, smallModel, vggAdapter,
    etc as defined below), this model adds fluke position prediction and whale
    ID prediction softmaxes.
    """
    def __init__(self, params):
        self.initializeModel(params)

    def name(self):
        prefix = ""
        if 'name' in self.params:
            prefix = self.params['name'] +"_"
        return prefix+"HumpbackGrayscaleTest"
    
    def preprocessX(self, X):
        """ Convert to rgb or grayscale if necessary """
        if 'preprocessGrayscale' not in self.params and 'preprocessRGB' not in self.params:
            return X
        newX = []
        for x in X:
            if self.params['preprocessGrayscale']:
                newX.append(skimage.color.rgb2gray(x).reshape(self.params['inputShape']))
            elif self.params['preprocessRGB']:
                newX.append(skimage.color.gray2rgb(x).reshape(self.params['inputShape']))
        return np.array(newX)
        
    def preprocessY(self, Y):
        """
        For preprocessing the target values (fluke positions, whale ID), we
        quantize any continuous values into params['nBins'] bins
        """
        regression_targets, id_targets = separateHumpbackLabels(Y, self.params['maxWhaleId'])
        targets = []
        for j in range(len(regression_targets[0])):
            targets.append(np.array(map(lambda x: quantizeValue(x[j], self.params['nBins']), regression_targets)))

        targets.append(np.array(id_targets))
        return targets
    
    def initializeModel(self, params):
        self.params = params

        modelFun = None
        modelInput = lastLayer = None
        if self.params['model'] == 'bigModel2':
            (modelInput, lastLayer) = bigModel2(self.params['inputShape'], self.params['partialStoredWeights'], self.params['partialWeightModelDepth'])
        if self.params['model'] == 'smallModel':
            (modelInput, lastLayer) = smallModel(self.params['inputShape'])     
        if self.params['model'] == 'vggAdapter':
            (modelInput, lastLayer) = vggAdapter(self.params['inputShape'])     
            
        # Much smaller fcc layer before final classification output for
        # memory considerations
        intermediate = Dense(self.params['denseSize'], activation="relu", init='he_normal')(lastLayer)
        
        # We need four outputs (fluke left x, left y, right x, right y, whale ID)
        outputs = [
            Dense(params['nBins'], activation='softmax', name='output_0', init='he_normal')(lastLayer),
            Dense(params['nBins'], activation='softmax', name='output_1', init='he_normal')(lastLayer),
            Dense(params['nBins'], activation='softmax', name='output_2', init='he_normal')(lastLayer),
            Dense(params['nBins'], activation='softmax', name='output_3', init='he_normal')(lastLayer),
            Dense(params['maxWhaleId']+1, activation='softmax', name='output_ids', init='he_normal')(intermediate)
        ]
        self.model = Model(input=[modelInput], output=outputs)

        # LR = 1.0 either converged quickly or never converged
        adadelta = Adadelta(lr=0.1)
        self.model.compile(loss="categorical_crossentropy", optimizer=adadelta)
        
    def evaluate(self, X_test, Y_test):
        """
        Evaluate model performance on the given heldout X_test data and Y_test labels
        """
        outputs = self.predict(X_test)

        regression_targets, id_targets = separateHumpbackLabels(Y_test, self.params['maxWhaleId'])
        targets = []
        for j in range(len(regression_targets[0])):
            targets.append(np.array(map(lambda x: quantizeValue(x[j], self.params['nBins']), regression_targets)))

        correct = 0
        correctIDs = 0
        correctIDPositions = []
        for j in range(len(outputs)):
            predictions = outputs[j]
            for i in range(len(predictions)):
                prediction = softmaxToCategory(predictions[i])
                if j == 4:
                    # Determine whether predicted ID was correct
                    target = softmaxToCategory(id_targets[i])
                    if prediction == target:
                        correctIDs += 1

                    # Determine where predicted ID is in list of classes
                    classes = classOutput(predictions[i])
                    pos = -1
                    for i in range(len(classes)):
                        if int(classes[i][0]) == int(target):
                            pos = i
                    correctIDPositions.append(pos)
                else:
                    target = softmaxToCategory(targets[j][i])
                    if prediction == target:
                        correct += 1
        print("Number correct fluke positions: "+str(correct) + " / "+str(len(X_test) * (len(outputs)-1)))
        print("Number correct whale IDs: "+str(correctIDs) + " / "+str(len(X_test) ))
        avgPos = float(sum(correctIDPositions)) / float(len(correctIDPositions))
        accuracy = avgPos / float(len(outputs[4][0]))
        print("Average correct whale ID position: "+str(avgPos)+"/"+str(len(outputs[4][0])))
        return {
            "accuracy": accuracy
        }

class VGGModel(CustomModel):
    """
    Used inside VGGDatagen to generate VGGNet output
    """
    def name(self):
        return "VGGModel"

    def initializeModel(self, params):
        h5f = h5py.File(params['h5weights'])
        inputLayer, outputLayer = vggModel.vggModel(h5f, params['modelDepth'])
        h5f.close()
        self.model = Model(input=[inputLayer], output=[outputLayer])
        adadelta = Adadelta(lr=1.0)
        self.model.compile(loss="categorical_crossentropy", optimizer=adadelta)

class VGGEnsemble(CustomModel):
    """
    This model expects its input to be the output of VGGModel above.
    Instead of learning one medium-sized net atop the VGG output, this model
      learns many small-sized nets and averages over their predictions.

    Since we cannot train nor evaluate all models simultaneously due to
    memory constraints, this class serializes model weights to disk and reloads 
    the models as necessary. 
    """
    models = []

    # Keep track of the index of the subset of models we're currently
    #  training simultaneously.
    activeSubsetIdx = 0

    # Keep track of the number of instances that the active subset of models
    # has been trained on, so we know when to swap them out for fresh models
    activeSubsetInstancesTrained = 0
    
    def name(self):
        return "VGGEnsemble"

    def loadWeights(self, annotation):
        for i in range(self.params['trainingConcurrency']):
            modelIdent = str(i + self.activeSubsetIdx)
            filepath = self.params["folderName"]+"/"+self.name()+"_model_"+modelIdent+".h5"
            self.models[i] = self.freshModel()
            if os.path.exists(filepath):
                print("Loading weights for ensemble model "+modelIdent+".")
                self.models[i].load_weights(filepath)
            else:
                print("No weights for ensemble model "+modelIdent+". Starting fresh.")

    def saveWeights(self, annotation):
        for i in range(self.params['trainingConcurrency']):
            modelIdent = str(i + self.activeSubsetIdx)
            self.models[i].save_weights(self.params["folderName"]+"/"+self.name()+"_model_"+modelIdent+".h5", overwrite=True)
    
    def initializeModel(self, params):
        self.params = params
        if "activeSubsetIdx" in params:
            self.activeSubsetIdx = params['activeSubsetIdx']
        for i in range(self.params['trainingConcurrency']):
            self.models.append(self.freshModel())
        self.loadWeights("")

    def freshModel(self):
        return miniVGGAdapter(self.params['inputShape'], self.params['denseSize'], self.params['nBins'], self.params['maxWhaleID'])
        
    def preprocessY(self, Y):
        regression_targets, id_targets = separateHumpbackLabels(Y, self.params['maxWhaleID'])
        targets = []
        for j in range(len(regression_targets[0])):
            targets.append(np.array(map(lambda x: quantizeValue(x[j], self.params['nBins']), regression_targets)))

        targets.append(np.array(id_targets))
        return targets

    def swapActiveSubset(self):
        """
        Swaps out the current active subset of models for others
        """
        self.saveWeights(self.params["folderName"])
        self.activeSubsetInstancesTrained = 0
        self.activeSubsetIdx = (self.activeSubsetIdx + self.params["trainingConcurrency"]) % self.params["ensembleSize"]
        print("Swapping out active subset. New subset idx: "+str(self.activeSubsetIdx))
        self.loadWeights(self.params["folderName"])
        
    def trainOnBatch(self, X_train, Y_train):
        if self.activeSubsetInstancesTrained > self.params['trainingInstancesPerModel']:
            self.swapActiveSubset()
        self.activeSubsetInstancesTrained += len(X_train)
        losses = []
        preprocessedY = self.preprocessY(Y_train)
        for model in self.models:
            losses.append(model.train_on_batch(self.preprocessX(X_train),
                                               preprocessedY))
        return float(sum(losses)) / float(len(losses))

    def evaluate(self, X_test, Y_test, wholeEnsemble=False):
        print("Evaluating....")
        outputs = []
        if not wholeEnsemble:
            outputs = self.predict(X_test)
        else:
            outputs = self.ensemblePredict(X_test)

        regression_targets, id_targets = separateHumpbackLabels(Y_test, self.params['maxWhaleID'])
        targets = []
        for j in range(len(regression_targets[0])):
            targets.append(np.array(map(lambda x: quantizeValue(x[j], self.params['nBins']), regression_targets)))

        correct = 0
        correctIDs = 0
        correctIDPositions = []
        for j in range(len(outputs)):
            predictions = outputs[j]
            for i in range(len(predictions)):
                prediction = softmaxToCategory(predictions[i])
                if j == 4:
                    # Determine whether predicted ID was correct
                    target = softmaxToCategory(id_targets[i])
                    if prediction == target:
                        correctIDs += 1

                    # Determine where predicted ID is in list of classes
                    classes = classOutput(predictions[i])
                    pos = -1
                    for i in range(len(classes)):
                        if int(classes[i][0]) == int(target):
                            pos = i
                    correctIDPositions.append(pos)
                else:
                    target = softmaxToCategory(targets[j][i])
                    if prediction == target:
                        correct += 1
        print("N correct: "+str(correct) + " / "+str(len(X_test) * (len(outputs)-1)))
        print("N classes correct: "+str(correctIDs) + " / "+str(len(X_test) ))
        avgPos = float(sum(correctIDPositions)) / float(len(correctIDPositions))
        print("Average correct class position: "+str(avgPos)+"/"+str(len(outputs[4][0])))
            
    def predict(self, X_test):
        """
        Predict by averaging over the softmax outputs.
        This function predicts over the current active subset of models.
        """
        outputs = []
        for model in self.models:
            outputs.append(model.predict(self.preprocessX(X_test)))

        averagedPredictions = []
        for outputTypeIdx in range(len(outputs[0])):
            predictions = []
            for modelOutput in outputs:
                predictions.append(modelOutput[outputTypeIdx])
            averagedPredictions.append(sum(predictions) / len(predictions))
        return averagedPredictions

    def ensemblePredict(self, X_test):
        """
        Use the entire ensemble to predict labels for the given instances.
        This is expensive, and will probably require loading models from disk. 
        """
        oldActiveSubsetIdx = self.activeSubsetIdx
        outputs = []
        for i in range(self.params['ensembleSize']):
            if i % self.params['trainingConcurrency'] == 0:
                self.activeSubsetIdx = i
                self.loadWeights("")
                outputs.append(self.predict(X_test))

        averagedPredictions = []
        for outputTypeIdx in range(len(outputs[0])):
            predictions = []
            for modelOutput in outputs:
                predictions.append(modelOutput[outputTypeIdx])
            averagedPredictions.append(sum(predictions) / len(predictions))
            
        self.activeSubsetIdx = oldActiveSubsetIdx
        return averagedPredictions
            

def chainLayers(layers, firstLayer):
    """
    Helper function to chain sequential layers together using keras functional api
    """
    lastLayer = firstLayer
    for layer in layers:
        lastLayer = layer(lastLayer)
    return lastLayer
        
def smallModel(input_shape):
    """
    Returns a root CNN model, on top of which we can build others with multiple outputs
    """
    print("Creating input with shape: "+str(input_shape))
    
    # Add our image shaped input
    modelInput = Input(name="image_input", shape=input_shape)

    # Setup our layers
    layers = [
        Convolution2D(32, 3, 3, border_mode='same', input_shape=input_shape), Activation('relu'), 
        Convolution2D(32, 3, 3), Activation('relu'), 
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5), 
        
        Convolution2D(64, 3, 3, border_mode='same'), Activation('relu'), 
        Convolution2D(64, 3, 3), Activation('relu'), 
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5), 

        Convolution2D(64, 3, 3, border_mode='same'), Activation('relu'), 
        Convolution2D(64, 3, 3), Activation('relu'), 
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),
        
        Flatten()
    ]
    
    lastLayer = chainLayers(layers, modelInput)
    return (modelInput, lastLayer)

def miniVGGAdapter(input_shape, denseSize, nBins, maxWhaleID):
    """
    Returns a small, precompiled mini network to train as part of an ensemble atop VGGnet
    """

    print("Initializing miniVGGAdapter with input shape: "+str(input_shape))

    # Add our image shaped input
    modelInput = Input(name="image_input_minivggadapter", shape=input_shape)
    
    # Setup our layers
    layers = [
        Convolution2D(32, 3, 3, border_mode='same', input_shape=input_shape), Activation('relu'), 
        Convolution2D(32, 3, 3), Activation('relu'), 
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5), 
        
        Convolution2D(64, 3, 3, border_mode='same'), Activation('relu'), 
        Convolution2D(64, 3, 3), Activation('relu'), 
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5), 
        Flatten(),
    ]

    lastLayer = chainLayers(layers, modelInput)
    intermediate = Dense(256, activation="relu")(lastLayer)
    
    # We need four outputs
    outputs = [
        Dense(nBins, activation='softmax', name='output_0')(lastLayer),
        Dense(nBins, activation='softmax', name='output_1')(lastLayer),
        Dense(nBins, activation='softmax', name='output_2')(lastLayer),
        Dense(nBins, activation='softmax', name='output_3')(lastLayer),
        Dense(maxWhaleID+1, activation='softmax', name='output_ids')(intermediate)
    ]
    model = Model(input=[modelInput], output=outputs)
        
    adadelta = Adadelta(lr=0.1)
    model.compile(loss="categorical_crossentropy", optimizer=adadelta)
    return model

def vggAdapter(input_shape):
    """
    Returns a small 2 layer convolutional net for use atop VGGNet
    """
    print("Creating VGG Adapter with input shape: "+str(input_shape))
    
    # Add our image shaped input
    modelInput = Input(name="image_input_vggadapter", shape=input_shape)
    
    # Setup our layers
    layers = [
        Convolution2D(64, 3, 3, init='he_normal', border_mode='same', input_shape=input_shape), Activation('relu'),
        Convolution2D(64, 3, 3, init='he_normal',input_shape=input_shape), Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        #Dropout(0.5),
        
        Convolution2D(256, 3, 3, init='he_normal', border_mode='same'), Activation('relu'),
        Convolution2D(256, 3, 3, init='he_normal'), Activation('relu'),
        Convolution2D(256, 3, 3, init='he_normal'), Activation('relu'), 
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5), 
        Flatten()
    ]

    lastLayer = chainLayers(layers, modelInput)
    return (modelInput, lastLayer)

def bigModel(input_shape):
    """
    Returns a large CNN model
    """
    print("Creating input with shape: "+str(input_shape))
    
    # Add our image shaped input
    modelInput = Input(name="image_input", shape=input_shape)
    
    # Setup our layers
    layers = [
        Convolution2D(64, 3, 3, border_mode='same', input_shape=input_shape), Activation('relu'),
        Convolution2D(64, 3, 3, input_shape=input_shape), Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Convolution2D(256, 3, 3, border_mode='same'), Activation('relu'),
        Convolution2D(256, 3, 3), Activation('relu'),
        Convolution2D(256, 3, 3), Activation('relu'), 
        MaxPooling2D(pool_size=(2, 2)),

        Convolution2D(256, 3, 3, border_mode='same'), Activation('relu'),
        Convolution2D(256, 3, 3), Activation('relu'),
        Convolution2D(256, 3, 3), Activation('relu'), 
        MaxPooling2D(pool_size=(2, 2)),
        
        Convolution2D(256, 3, 3, border_mode='same'), Activation('relu'),
        Convolution2D(256, 3, 3), Activation('relu'),
        Convolution2D(256, 3, 3), Activation('relu'), 
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5), 

        Flatten()
    ]

    lastLayer = chainLayers(layers, modelInput)
    return (modelInput, lastLayer)

def layerWeights(h5Model):
    """
    Track the number of layers of each type we have in our model.
    In this way, we can restore weights from a standard saved keras model .h5 file
    even when our topology is larger than that of the stored file. 
    """
    layerCount = {}
    def initialWeights(ty):
        if ty not in layerCount:
            layerCount[ty] = 0
        layerCount[ty] += 1
        name = ty+"_"+str(layerCount[ty])

        if h5Model is None:
            print("No presaved weights for layer "+name)
            return None
        
        if name in h5Model:
            #print("---- Loading presaved weights for layer "+name)
            return [h5Model[name][name+"_W:0"], h5Model[name][name+"_b:0"]]
        else:
            print("Failed to load presaved weights for layer "+name)
        return None
    return initialWeights

def bigModel2(input_shape, h5Model=None, modelDepth=None):
    """
    Returns a larger CNN model

    If provided both a keras model import and a model depth 
    given via `h5model` and `modelDepth`, then the first `modelDepth`
    convolutional layers will be reloaded from the weight file
    """
    print("Creating input with shape: "+str(input_shape))
    print("Model depth: "+str(modelDepth))
    
    # Add our image shaped input
    modelInput = Input(name="image_input", shape=input_shape)

    initialWeights = layerWeights(h5Model)
    
    # Setup our layers
    layers = [
        Convolution2D(64, 3, 3, border_mode='same', init='he_normal', input_shape=input_shape, weights=initialWeights("convolution2d")), Activation('relu'),
        Convolution2D(64, 3, 3, border_mode='same', init='he_normal', weights=initialWeights("convolution2d")), Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        #Dropout(0.5),
    ]
    if modelDepth == None or modelDepth >= 2:
        layers = layers + [
            Convolution2D(128, 3, 3, border_mode='same', init='he_normal', weights=initialWeights("convolution2d")), Activation('relu'),
            Convolution2D(128, 3, 3, border_mode='same', init='he_normal', weights=initialWeights("convolution2d")), Activation('relu'), 
            MaxPooling2D(pool_size=(2, 2)),
            #Dropout(0.5),
        ]
    if modelDepth == None or modelDepth >= 3:
        layers = layers + [
            Convolution2D(256, 3, 3, border_mode='same', init='he_normal', weights=initialWeights("convolution2d")), Activation('relu'),
            Convolution2D(256, 3, 3, border_mode='same', init='he_normal', weights=initialWeights("convolution2d")), Activation('relu'),
            Convolution2D(256, 3, 3, border_mode='same', init='he_normal', weights=initialWeights("convolution2d")), Activation('relu'), 
            MaxPooling2D(pool_size=(2, 2)),
            #Dropout(0.5), 
        ]

    if modelDepth == None or modelDepth >= 4:
        layers = layers + [
            Convolution2D(256, 3, 3, border_mode='same', init='he_normal', weights=initialWeights("convolution2d")), Activation('relu'),
            Convolution2D(256, 3, 3, border_mode='same', init='he_normal', weights=initialWeights("convolution2d")), Activation('relu'),
            Convolution2D(256, 3, 3, border_mode='same', init='he_normal', weights=initialWeights("convolution2d")), Activation('relu'), 
            MaxPooling2D(pool_size=(2, 2)),
            #Dropout(0.5), 
        ]
        

    layers = layers + [
        MaxPooling2D(pool_size=(2, 2)),        
        Dropout(0.5), 
        Flatten()
    ]

    lastLayer = chainLayers(layers, modelInput)
    return (modelInput, lastLayer)

def softmaxToCategory(softmaxOutput):
    """ Converts an array of softmax output into a predicted category"""
    maxIdx = -1
    maxV = -1
    for i in range(len(softmaxOutput)):
        if softmaxOutput[i] > maxV:
            maxV = softmaxOutput[i]
            maxIdx = i
    return maxIdx

def separateHumpbackLabels(labels, maxWhaleID):
    """
    Helper function to take a given row of data 
    (combined whaleIDs and regression targets), 
    and to return separate (regression_targets, one_hot_id_targets)
    """
    regressionTargets = []
    oneHotIDTargets = []
    for label in labels:
        if len(label) < 5:
            continue
        regressionTargets.append(label[1:5]) # Exclude whale ID & bounding box info
        oneHot = np.zeros(maxWhaleID+1)
        oneHot[int(label[0])] = 1.0
        oneHotIDTargets.append(oneHot)
    return np.array(regressionTargets), np.array(oneHotIDTargets)
