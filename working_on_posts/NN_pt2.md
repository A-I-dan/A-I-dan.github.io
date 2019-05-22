---
layout:     post
title:      "The Math: Why a Neural Network Works"
date:       2019-05-2 12:00:00
author:     "A.I. Dan"
---

# The Math Behind Simple Neural Nets

In this post I will be referencing my earlier post, titled ["Simple Neural Networks With Numpy"](https://a-i-dan.github.io/tanh_NN) and I will be going more in depth with the math that makes the neural network actually function properly.

I will start by showing the example code along with the results it produces.

```python
import numpy as np

# create the tanh function with an optional derivative
def tanh(x, deriv = False):
    if deriv == True:
        return (1 - (tanh(np.exp2(2) * x)))
    return (np.exp(2*x) - 1) / (np.exp(2*x) + 1)

# create dataset
input_layer = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])   
outputs = np.array([[0, 0, 1, 1]]).T

np.random.seed(1)

# itialize the random synaptic weights
synapse = np.random.random((3, 1)) - 1

# training process
for iter in range(10000):
    hidden_layer = tanh(np.dot(input_layer, synapse))
    hidden_error = outputs - hidden_layer
    hidden_delta = hidden_error * tanh(hidden_layer, True)
    synapse += np.dot(input_layer.T, hidden_delta)

# prediction for new inputs  
def predict(new_input):
    return tanh(np.dot(new_input, synapse))

# new test data (examples not used in training dataset)
test_data = np.array([ [1,1,0],
                       [0,1,0],
                       [1,0,0] ])

# print predicted results for each new test example
for test_example in test_data:
        print('Predicted Output For:', test_example,'=', predict(test_example), '\n')
```
Output:
```
Predicted Output For: [1 1 0] = [0.99999874]

Predicted Output For: [0 1 0] = [-0.01759575]

Predicted Output For: [1 0 0] = [0.99999878]
```
