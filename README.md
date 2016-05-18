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
docker run -it -v -p 5000:5000 /path/to/this/repo/:/humpbackModel/ humpbackModel
```

Then, you should be able to run the model inside the container:
```
python /humpbackModel/server.py
```

## Running the prediction server
You can run the prediction server with:

```
python /humpbackModel/server.py
```
This will start a simple flask server on port 5000 that will perform predictions using any of the available experiment classes defined in `experiments.py`

To perform predictions, at least two files must be present:
  - `experiment_name`_params.json
  - experiments/`experiment_name`/trials.h5

Additionally, any files needed by the particular experiment (as given by `experiment_name`_params.json) must be present. 

To make a prediction using the server, you can make a request to 

```
127.0.0.1:5000/algorithm/<experiment_name>/predictImage?filename=file
```

Where filename is the path to a file from the CWD of server.py.


## Usage of manage.py
Usage of manage.py can also be obtained by ```python manage.py -h```

### Prediction
To predict the identity of a whale in an image, you can run:
```
python manage.py --experiment <experiment_name> --predict <filename> {... experiment specific arguments ...}
```

Specifically for VGGStack, you can run:
```
python manage.py --experiment VGGStack --predict orig25/10419-nb99-2734-27.jpg --vggWeights VGG/FullVGGWeights.h5 --dataTrain humpbackClassification_trainingData.json --dataTest humpbackClassification_testingData.json --imageFolder orig25/
```

### Training
To train VGGStack, you can run:
```
python manage.py --experiment VGGStack --train --vggWeights VGG/FullVGGWeights.h5 --dataTrain humpbackClassification_trainingData.json --dataTest humpbackClassification_testingData.json --imageFolder orig25/
```

### Evaluation
To guage the accuracy of VGGStack on the held out test set:
```
python manage.py --experiment VGGStack --evaluate --vggWeights VGG/FullVGGWeights.h5 --dataTrain humpbackClassification_trainingData.json --dataTest humpbackClassification_testingData.json --imageFolder orig25/
```

### Training
To train VGGStack, you'd run:
```
python manage.py --experiment VGGStack --train --vggWeights VGG/FullVGGWeights.h5 --dataTrain humpbackClassification_trainingData.json --dataTest humpbackClassification_testingData.json --imageFolder orig25/
```

## Additional Files
The default model in this project uses the first several layers of VGGNet for feature extraction. The script expects a saved keras VGGNet model file to be located at VGG/FullVGGWeights.h5. It's not included in this repository as the weights are over 500mb. Contact me if you need them - I'll host them publically at some point soon. 

## Generating training data
To generate trainingData.csv from a new batch of images, you can use the preprocessing/humpbackData.py script:
```
python preprocessing/humpbackData.py <zooniverse file> <image directory>
```

Note that this new dataset will contain a different number of whales (and in a different order) than you  had when training previous models, so their stored weights will not be compatible with the new data set. 

This will be resolved soon, and only the final fully connected layer will need to be retrained when given new whales to identify.
