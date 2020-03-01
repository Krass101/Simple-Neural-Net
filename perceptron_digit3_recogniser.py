from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np 

# Use SKLearn digits dataset
digits = datasets.load_digits()

# Create feature matrix
digits_set = digits.data

# Sigmoid and sigmoid derivitive functions for normalising outputs
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivitive(x):
    return x * (1-x)

# Create an array for the training input, which will be 16 '3' digits and 16 'non-3' digits.
training_input = []

# Array elements at index places ending in 3 in the SKLearn set are hand written 3's. E.g. index 3, 13, 23..etc. are all 3's while index 10,20,30..etc. are 0's.
# So here we make a list of 3's and 0's from the SKLearn set which will serve as a trainig set
training_digits = [3,10,13,20,23,30,33,40,43,50,53,60,63,70,73,80,83,90,93,100,103,110,113,120,123,130,133,140,143,150,153,160]

# We use a loop to collect the 3's and 0's from SKLearn and place them into the training_input array
i = 0
while i < len(training_digits):    
    training_input.append(digits_set[training_digits[i]])
    i += 1

# Convert training_input into array, from a list beacuse lists cannot be transposed which is something we need to do when working with matrices 
training_input = np.array(training_input)

# We need to normalise the data in the arrays to be between 0 and 1. 
def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom 

training_input_scaled = scale(training_input, 0, 1)

# Set up the starting synaptic weights. In this case we simply make them all 0.1 to start with. We need 64 synapses because our data is an 8x8 image which gets split into 64 'pixels'
synaptic_weights = np.array([[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]]).T

# Identify which training inputs are the digit 3 and which are not. 1 means it's a '3' and 0 means it is not a '3'
training_outputs = np.array([[1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,]]).T

# Running the perceptron and adjusting the synaptic weights 
for itteration in range(2000000):
    input_layer = training_input_scaled
    output = sigmoid(np.dot(input_layer, synaptic_weights))
    error = training_outputs - output
    adjustments = error * sigmoid_derivitive(output)
    synaptic_weights += np.dot(input_layer.T, adjustments)

# Print results to see what happened
# Synaptic weights have been adjusted successfully based on the back propagation 
print ('Synaptic_weights')
print (synaptic_weights)

# TESTING
# Test 5 examples of 3's which the perceptron has not seen before 
test_input_real_3_1 = np.array(digits_set[563])
test_output_real_3_1 = sigmoid(np.dot(test_input_real_3_1, synaptic_weights))
print ('Test output: test_input_real_3_1')
print (test_output_real_3_1)

test_input_real_3_2 = np.array(digits_set[573])
test_output_real_3_2 = sigmoid(np.dot(test_input_real_3_2, synaptic_weights))
print ('Test output: test_input_real_3_2')
print (test_output_real_3_2)

test_input_real_3_3 = np.array(digits_set[583])
test_output_real_3_3 = sigmoid(np.dot(test_input_real_3_3, synaptic_weights))
print ('Test output: test_input_real_3_3')
print (test_output_real_3_3)

test_input_real_3_4 = np.array(digits_set[593])
test_output_real_3_4 = sigmoid(np.dot(test_input_real_3_4, synaptic_weights))
print ('Test output: test_input_real_3_4')
print (test_output_real_3_4)

test_input_real_3_5 = np.array(digits_set[603])
test_output_real_3_5 = sigmoid(np.dot(test_input_real_3_5, synaptic_weights))
print ('Test output: test_input_real_3_5')
print (test_output_real_3_5)



# Test 5 examples of non-3's which the perceptron has not seen before 
test_input_non_3_1 = np.array(digits_set[700])
test_output_non_3_1 = sigmoid(np.dot(test_input_non_3_1, synaptic_weights))
print ('Test output: test_input_non_3_1')
print (test_output_non_3_1)

test_input_non_3_2 = np.array(digits_set[800])
test_output_non_3_2 = sigmoid(np.dot(test_input_non_3_2, synaptic_weights))
print ('Test output: test_input_non_3_2')
print (test_output_non_3_2)

test_input_non_3_3 = np.array(digits_set[900])
test_output_non_3_3 = sigmoid(np.dot(test_input_non_3_3, synaptic_weights))
print ('Test output: test_input_non_3_3')
print (test_output_non_3_3)

test_input_non_3_4 = np.array(digits_set[1000])
test_output_non_3_4 = sigmoid(np.dot(test_input_non_3_4, synaptic_weights))
print ('Test output: test_input_non_3_4')
print (test_output_non_3_4)

test_input_non_3_5 = np.array(digits_set[1100])
test_output_non_3_5 = sigmoid(np.dot(test_input_non_3_5, synaptic_weights))
print ('Test output: test_input_non_3_5')
print (test_output_non_3_5)