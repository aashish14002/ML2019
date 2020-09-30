# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 19:24:46 2019

@author: Aashish Kumar pcr902
"""
import numpy as np
import matplotlib.pyplot as plt

imageData = np.loadtxt("MNIST-Train-cropped.txt").reshape(10000, 784)
imageLabelData = np.loadtxt("MNIST-Train-Labels-cropped.txt",dtype=np.uint16)

imageTestData = np.loadtxt("MNIST-Test-cropped.txt").reshape(2000, 784)
imageTestLabelData = np.loadtxt("MNIST-Test-Labels-cropped.txt",dtype=np.uint16)

#print(imageLabelData[1])
#col1 = imageData[1].reshape(28,28)
#plt.imshow(np.transpose(col1), cmap="gray")
#plt.show()

kValues = np.arange(1,34,2)
[imageTrainData, imageValidationData] = [imageData[0:8000], imageData[8000:]]
[imageTrainLabelData, imageValidationLabelData] = [imageLabelData[0:8000], imageLabelData[8000:]]

zeroImageTrainData = np.array([imageTrainData[i] for i in range(imageTrainLabelData.size) if imageTrainLabelData[i] == 0])
oneImageTrainData =  np.array([imageTrainData[i] for i in range(imageTrainLabelData.size) if imageTrainLabelData[i] == 1])
fiveImageTrainData =  np.array([imageTrainData[i] for i in range(imageTrainLabelData.size) if imageTrainLabelData[i] == 5])
sixImageTrainData =  np.array([imageTrainData[i] for i in range(imageTrainLabelData.size) if imageTrainLabelData[i] == 6])
eightImageTrainData =  np.array([imageTrainData[i] for i in range(imageTrainLabelData.size) if imageTrainLabelData[i] == 8])

zeroImageValData = np.array([imageValidationData[i] for i in range(imageValidationLabelData.size) if imageValidationLabelData[i] == 0])
oneImageValData =  np.array([imageValidationData[i] for i in range(imageValidationLabelData.size) if imageValidationLabelData[i] == 1])
fiveImageValData =  np.array([imageValidationData[i] for i in range(imageValidationLabelData.size) if imageValidationLabelData[i] == 5])
sixImageValData =  np.array([imageValidationData[i] for i in range(imageValidationLabelData.size) if imageValidationLabelData[i] == 6])
eightImageValData =  np.array([imageValidationData[i] for i in range(imageValidationLabelData.size) if imageValidationLabelData[i] == 8])

zeroImageTestData = np.array([imageTestData[i] for i in range(imageTestLabelData.size) if imageTestLabelData[i] == 0])
oneImageTestData =  np.array([imageTestData[i] for i in range(imageTestLabelData.size) if imageTestLabelData[i] == 1])
fiveImageTesnData =  np.array([imageTestData[i] for i in range(imageTestLabelData.size) if imageTestLabelData[i] == 5])
sixImageTestData =  np.array([imageTestData[i] for i in range(imageTestLabelData.size) if imageTestLabelData[i] == 6])
eightImageTestData =  np.array([imageTestData[i] for i in range(imageTestLabelData.size) if imageTestLabelData[i] == 8])

        
zeroLabelCount =  zeroImageTrainData.size
oneLabelCount = oneImageTrainData.size
fiveLabelCount =  fiveImageTrainData.size
sixLabelCount = sixImageTrainData.size
eightLabelCount =  eightImageTrainData.size

plusOneArray = np.repeat([1],[zeroLabelCount])
minusOneArray = np.repeat([-1],[oneLabelCount])

validationError1 = np.zeros(33)
testError1 = np.zeros(33)

def merge_structured_arrays(array1, array2):
    n1 = len(array1)
    n2 = len(array2)
    array_out = array1.copy()
    array_out.resize(n1 + n2)
    array_out[n1:] = array2
    return array_out

_imageValidData = np.append(zeroImageValData, oneImageValData)
_imageValidLabel = np.repeat([1,-1],[zeroImageValData.size, oneImageValData.size])

_imageTestData = np.append(zeroImageTestData, oneImageTestData)
_imageTestLabel = np.repeat([1,-1],[zeroImageTestData.size, oneImageTestData.size])

_imageTrainData = np.append(zeroImageTrainData, oneImageTrainData)
_imageTrainLabel = np.repeat([1,-1],[zeroImageTrainData.size, oneImageTrainData.size])

_totalValidLabel = np.repeat(_imageValidLabel.size, 33)
_totalTestLabel = np.repeat(_imageTestLabel.size, 33)

for i in range(_imageValidData.size): 
    referenceValidDataMatrix = np.repeat(np.array([_imageValidData[i]]), [_imageTrainData.size], axis=0)
    referenceValidLabelMatrix = np.repeat(_imageValidLabel[i], 33)
    
    distanceValidationMatrix = np.sqrt(np.sum((referenceValidDataMatrix - _imageTrainData) ** 2, axis=1))
    
    dtype = np.dtype([('distance',distanceValidationMatrix.dtype),('label',_imageTrainLabel.dtype)])
    structDistanceValidationMatrix = np.empty(len(distanceValidationMatrix),dtype=dtype)
    structDistanceValidationMatrix['distance'] = distanceValidationMatrix
    structDistanceValidationMatrix['label'] = _imageTrainLabel
    
    validationMatrix = np.sort(structDistanceValidationMatrix,order='distance')
    cumValidationMatrix = np.cumsum(validationMatrix['distance'][0:33])
    KNNValid = np.add(cumValidationMatrix, referenceValidLabelMatrix)
    for j in range(33):
        if(KNNValid[j] == 0):
            validationError1[j] += 1
            
            
for i in range(_imageTestData.size): 
   referenceTestDataMatrix = np.repeat(np.array([imageTestData[i]]), [_imageTrainData.size], axis=0)
   referenceTestLabelMatrix = np.repeat(_imageTestLabel[i], 33)
   distanceTestMatrix = np.sqrt(np.sum((referenceTestDataMatrix - _imageTrainData) ** 2, axis=1))
   dtype = np.dtype([('distance',distanceTestMatrix.dtype),('label',_imageTrainLabel.dtype)])
   structDistanceTestMatrix = np.empty(len(distanceTestMatrix),dtype=dtype)
   structDistanceTestMatrix['distance'] = distanceTestMatrix
   structDistanceTestMatrix['label'] = _imageTrainLabel
    
   testMatrix = np.sort(structDistanceTestMatrix,order='distance')
   cumTestMatrix = np.cumsum(testMatrix['distance'][0:33])
   KNNTest = np.add(cumTestMatrix, referenceTestLabelMatrix)
   for j in range(33):
       if(KNNTest[j] == 0):
           testError1[j] += 1
           
ValidError = np.array([ i for i in range(np.divide(validationError1, _totalValidLabel)) if (i+1)%2!=0 ])
testError = np.array([ i for i in range(np.divide(testError1, _totalTestLabel)) if (i+1)%2!=0 ])

fig, ax = plt.subplots()
ax.plot(kValues,ValidError,label="Valid error")
ax.plot(kValues, testError, label="test error")
ax.legend(loc='upper right')

ax.set(xlabel='K values', ylabel='error',
       title='KNN for 0 and 1')

fig.savefig('Part4.png')

plt.show()

            
    
    
    
    
    
    
#    distanceFiveValidationMatrix = np.sqrt(np.sum((referenceValidDataMatrix - fiveImageTrainData) ** 2, axis=1))
#    distanceFiveTestMatrix = np.sqrt(np.sum((referenceTestDataMatrix - fiveImageTrainData) ** 2, axis=1))
#    
#    distanceEightValidationMatrix = np.sqrt(np.sum((referenceValidDataMatrix - eightImageTrainData) ** 2, axis=1))
#    distanceEightTestMatrix = np.sqrt(np.sum((referenceTestDataMatrix - eightImageTrainData) ** 2, axis=1))
    
    
