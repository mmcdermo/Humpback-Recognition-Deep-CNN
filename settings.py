import math

batchSize = 16
miniBatchSize = 32
nEpoch = 200
ratio = 7./4. # desired w/h ratio of images
imgRows, imgCols = int(math.floor(256 / ratio)), 256
imgChannels = 3
nBins = 40
