import numpy as np
import sys

sys.setrecursionlimit(1500000)

def prediction(inputs, weights):
    product = inputs * weights
    netValue = np.sum(product)
    print('Net Value: ', netValue)
    actualOutput = np.sign(netValue)
    return actualOutput


def changeWeights(inputs, prevWeights, c, desiredOutput, actualOutput, error):
    newWeights = prevWeights
    diff = desiredOutput - actualOutput
    print('Diff: ', diff)
    error = error + abs(diff)

    if actualOutput != desiredOutput:
        newWeights = prevWeights + c * diff * inputs

    return [newWeights, error]


dataInputs = np.array(
    [[1.0, 1.0], [9.4, 6.4], [2.5, 2.1],
     [8.0, 7.7], [0.5, 2.2], [7.9, 8.4],
     [7.0, 7.0], [2.8, 0.8], [1.2, 3.0],
     [7.8, 6.1]])

dataOutputs = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
weights = np.array([0.75, 0.5, -0.6])
bias = 1
c = 0.2
epoch = 1000

for i in range(0, epoch):
    print('################# START EPOCH #################', i)
    index = 0
    epoch = epoch + 1
    newDataOutputs = []
    error = 0
    for inputs in dataInputs:
        desiredOutput = dataOutputs[index]
        inputs = np.append(inputs, bias)
        actualOutput = prediction(inputs, weights)
        newDataOutputs.append(actualOutput)
        result = changeWeights(inputs, weights, c, desiredOutput, actualOutput, error)
        weights = result[0]
        error = result[1]
        print('Inputs: ', inputs, 'Desired Output: ', desiredOutput, 'Actual Output: ', actualOutput, 'Weights: ',
              weights, 'Error: ', error)
        print('==========')
        index = index + 1
    print('Data Output: ', dataOutputs)
    print('Epoch Output: ', newDataOutputs)
    newDataOutputs = []
    if error == 0:
        break
    print('################# END EPOCH #################', i)
