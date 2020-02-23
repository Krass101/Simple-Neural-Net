from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np 

# Use sklearn for digit data
digits = datasets.load_digits()
print (digits.images[43])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivitive(x):
    return x * (1-x)

# These are digits, normalised to be 0 or 1 values. The goal of the neural net is to identify the digit 3. 
training_input = np.array ([[ 0,  0,  0,  1, 1, 1,  0,  0, 0,  0,  1, 1, 1, 1,  0,  0, 0,  0,  1, 1, 1, 1,  0,  0, 0,  0,  1,  1, 1, 1,  0,  0, 0,  1,  1, 1, 1,  1,  0,  0, 0,  1,  1,  1,  1,  0,  0,  0, 0,  1,  1,  1,  1, 1,  1,  0, 0,  0,  0,  1, 1,  1,  1,  0],
                            [ 0,  0,  1,  1, 1, 1,  0,  0, 0,  1,  1, 1, 1, 1,  0,  0, 0,  1,  1, 1, 1, 0,  0,  0, 0,  0,  1,  1, 1, 1,  0,  0, 0,  0,  0, 1, 1,  1,  1,  0, 0,  0,  0,  0,  1,  1,  1,  0, 0,  0,  1,  1,  1, 1,  1,  0, 0,  0,  1,  1, 1,  1,  0,  0],
                            [ 0,  1,  1,  1, 1, 1,  1,  0, 0,  1,  1, 1, 1, 1,  1,  0, 0,  0,  0, 1, 1, 1,  1,  0, 0,  0,  0,  1, 1, 1,  0,  0, 0,  0,  0, 1, 1,  1,  0,  0, 0,  0,  0,  0,  1,  1,  1,  0, 0,  1,  1,  1,  1, 1,  1,  0, 0,  1,  1,  1, 1,  1,  0,  0],
                            [ 0,  0,  0,  1, 1, 1,  0,  0, 0,  0,  1, 1, 1, 1,  0,  0, 0,  0,  1, 1, 1, 1,  0,  0, 0,  0,  1,  1, 1, 1,  0,  0, 0,  1,  1, 1, 1,  1,  0,  0, 0,  1,  1,  1,  1,  0,  0,  0, 0,  1,  1,  1,  1, 1,  1,  0, 0,  0,  0,  1, 1,  1,  1,  0],
                            [ 0,  0,  0,  1, 1, 1,  0,  0, 0,  0,  1, 1, 1, 1,  0,  0, 0,  0,  1, 1, 1, 1,  0,  0, 0,  0,  1,  1, 1, 1,  0,  0, 0,  1,  1, 1, 1,  1,  0,  0, 0,  1,  1,  1,  1,  0,  0,  0, 0,  1,  1,  1,  1, 1,  1,  0, 0,  0,  0,  1, 1,  1,  1,  0]])

# Placeholder starting weights
synaptic_weights = np.array([[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]]).T

# Legend identifying which training inputs are the digit 3 and which are not
training_outputs = np.array([[0,1,1,0,0]]).T

# Running the perceptron and adjusting the weights 
for itteration in range(200000):
    input_layer = training_input
    output = sigmoid(np.dot(input_layer, synaptic_weights))
    error = training_outputs - output
    adjustments = error * sigmoid_derivitive(output)
    synaptic_weights += np.dot(input_layer.T, adjustments)

# Print output to see what happens
print ('training input')
print (training_input)

print ('Output')
print (output)

print ('sigmoid deriv')
print (sigmoid_derivitive(output))

print ('error')
print (error)

print ('adjustments')
print (adjustments)

print ('adjustoments to weights')
print (np.dot(input_layer.T, adjustments))

print ('input layer T')
print (input_layer.T)

print ('synaptic_weights')
print (synaptic_weights)

# Test example representin the digit 3
test_input_three = np.array ([[ 0,  0,  0,  1, 1, 1,  0,  0, 0,  0,  1,  1,  1, 1,  0,  0, 0,  0,  0,  0,  0, 1,  1,  0, 0,  1,  1,  1, 1, 1,  1,  0, 0,  1, 1, 1, 1, 1,  1,  0, 0,  0,  0,  0, 1,  1,  0,  0, 0,  0,  0,  1, 1,  0,  0,  0, 0,  0,  0, 1,  1,  0,  0,  0,] ])

# Expectation is that it will be close to 1 if it can recognise the digit 3 above
test_output = sigmoid(np.dot(test_input_three, synaptic_weights))

print ('test output')
print (test_output)