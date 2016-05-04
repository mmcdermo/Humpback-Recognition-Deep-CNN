from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D

def chainLayers(layers, firstLayer):
    """
    Helper function to chain sequential layers together using keras functional api
    """
    lastLayer = firstLayer
    i = 0
    for layer in layers:
        i += 1
        lastLayer = layer(lastLayer)
    return lastLayer

def vggModel(h5Model=None, modelDepth=8):
    """
    Returns a subset of VGGNet, using pretrained layers from the h5model
    """
    input_shape = (3, 224, 224)
    print("Generating a VGG Model")
    print("Creating input with shape: "+str(input_shape))
    print("Model depth: "+str(modelDepth))
    
    # Add our image shaped input
    modelInput = Input(name="image_input", shape=input_shape)

    # Track the number of layers of each type we have in our model.
    #  In this way, we can restore weights from a standard saved keras model .h5 file
    #  even when our topology is larger than that of the stored file. 
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
            print("---- Loading presaved weights for layer "+name)
            return [h5Model[name][name+"_W"], h5Model[name][name+"_b"]]
        else:
            print("Failed to load presaved weights for layer "+name)
        return None

    layers = [
        Convolution2D(64, 3, 3, border_mode='same', activation='relu', weights=initialWeights("convolution2d"), input_shape=input_shape),
        Convolution2D(64, 3, 3, border_mode='same', activation='relu', weights=initialWeights("convolution2d")),
        MaxPooling2D(pool_size=(2, 2))
    ]
    if modelDepth >= 2:
        layers += [
            Convolution2D(128, 3, 3, border_mode='same', activation='relu', weights=initialWeights("convolution2d")),
            Convolution2D(128, 3, 3, border_mode='same', activation='relu', weights=initialWeights("convolution2d")),
            MaxPooling2D(pool_size=(2, 2))
        ]

    if modelDepth >= 3:
        layers += [
            Convolution2D(256, 3, 3, border_mode='same', activation='relu', weights=initialWeights("convolution2d")),
            Convolution2D(256, 3, 3, border_mode='same', activation='relu', weights=initialWeights("convolution2d")),
            Convolution2D(256, 3, 3, border_mode='same', activation='relu', weights=initialWeights("convolution2d")),
            MaxPooling2D(pool_size=(2, 2))
        ]
    
    if modelDepth >= 4:
        layers += [
            Convolution2D(512, 3, 3, border_mode='same', activation='relu', weights=initialWeights("convolution2d")),
            Convolution2D(512, 3, 3, border_mode='same', activation='relu', weights=initialWeights("convolution2d")),
            Convolution2D(512, 3, 3, border_mode='same', activation='relu', weights=initialWeights("convolution2d")),
            MaxPooling2D(pool_size=(2, 2))
        ]
        
    if modelDepth >= 5:
        layers += [
            Convolution2D(512, 3, 3, border_mode='same', activation='relu', weights=initialWeights("convolution2d")),
            Convolution2D(512, 3, 3, border_mode='same', activation='relu', weights=initialWeights("convolution2d")),
            Convolution2D(512, 3, 3, border_mode='same', activation='relu', weights=initialWeights("convolution2d")),
            MaxPooling2D(pool_size=(2, 2))
        ]
    if modelDepth >= 6:
        layers += [
            Flatten(),
            Dense(4096, weights=initialWeights("dense")),
            Activation('relu'),
            Dropout(0.5)
        ]
    if modelDepth >= 7:
        layers += [
            Dense(4096, weights=initialWeights("dense")),
            Activation('relu'),
            Dropout(0.5) 
        ]
    if modelDepth >= 8:
        layers += [
            Dense(1000, weights=initialWeights("dense")),
            model.add(Activation('softmax'))
        ]

    lastLayer = chainLayers(layers, modelInput)
    return (modelInput, lastLayer)
