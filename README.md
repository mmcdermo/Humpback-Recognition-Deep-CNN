# Humpback Recognition using deep CNNs
This project uses deep convolutional neural networks to predict whale identities from images of their flukes. Currently, the model requires images to be precropped.

## Docker Instructions
For ease of use, this repo includes a dockerfile at docker/Dockerfile that can be used to run the model. 

If you'd like to run the model in CPU-only mode, change the first line in the Dockerfile to 
```
FROM gcr.io/tensorflow/tensorflow
```

Then, you'll need to build the docker image (installing dependencies etc may take a little while):
```
docker build -t humpbackModel ./Docker
```

Once the docker image is built, you can run it and mount this github repo inside of it:
```
docker run -it /path/to/this/repo/:/humpbackModel/ humpbackModel
```

Then, you should be able to run the model inside the container:
```
/humpbackModel/run.py -h
```

## Usage and examples
Usage of run.py can also be obtained by run.py -h

### Prediction
To predict the identity of a whale in an image, you can run:
```
python run.py --trainCSV trainingData.csv --imageDir imageDirectory/ --predict whale-fluke-image.jpg --storedWeights weightsFile.h5
```

### Evaluation
To guage the accuracy of the model on the held out test set:
```
python run.py --trainCSV trainingData.csv --imageDir imageDirectory/ --evaluate --storedWeights weightsFile.h5
```

### Training
To train a model from scratch:
```
python run.py --trainCSV trainingData.csv --imageDir imageDirectory/ --train
```

## Additional Files
The default model in this project uses a pretrained VGGNet deep CNN as a feature extraction stage. The script expects this file to be located at VGG/FullVGGWeights.h5. It's not included in this repository as the weights are over 500mb. Contact me if you need them - I'll host them publically at some point soon. 

## Generating training data
To generate trainingData.csv from a new batch of images, you can use the preprocessing/humpbackData.py script:
```
python preprocessing/humpbackData.py <zooniverse file> <image directory>
```

Note that this new dataset will contain a different number of whales (and in a different order) than you  had when training previous models, so their stored weights will not be compatible with the new data set. 

This will be resolved soon, and only the final fully connected layer will need to be retrained when given new whales to identify.
