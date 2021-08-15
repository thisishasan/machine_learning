import numpy as np
import math

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = [0, 1, 1, 0]
rate = 0.8
noOfNodes = 3
weights = np.array([[0.2, 0.4], [-0.3, 0.1], [-0.3, -0.2]])
bias = np.array([-0.4, 0.2, 0.1])
threshold = 1

for input in inputs:
    lastNodeInputs = []
    nodeOutputs = []
    netInputs = []
    lastNodeError = 0
    lastNodeIndex = 0
    errors = []
    for i in range(0, noOfNodes):
        if (i < 2):
            product = weights[i] * input
        else:
            product = weights[i] * np.array(lastNodeInputs)
            lastNodeIndex = i
        netInput = np.sum(product) + bias[i]
        netInputs.append(netInput)
        output = 1 / (1 + math.exp(-1 * netInput))
        lastNodeInputs.append(output)
        nodeOutputs.append(output)

    for i in range(noOfNodes - 1, -1, -1):
        if (i == 2):
            lastNodeError = nodeOutputs[i] * (1 - nodeOutputs[i]) * (threshold - nodeOutputs[i])
            errors.append(lastNodeError)
        else:
            error = nodeOutputs[i] * (1 - nodeOutputs[i]) * lastNodeError * weights[lastNodeIndex][i]
            errors.append(error)

    print('net inputs', netInputs)
    print('outputs', nodeOutputs)
    print('errors', errors)
    print('#######################################')
