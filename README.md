# Perceptron that recognizes the digit 3

A simple one-layer perceptron, which can tell if a hand written image is a 3 or not.

## Detils

Goal: Build 'as simple as possible' perceptron.

Data: KLearn digits library which is a library of 8x8 hand written images.

Process: One-layer perceptron with back propagation chosen as the simplest possible implementation of a 'Neural Network'. The maths for calculating synaptic weights, sigmoid, and errors are all written from scratch and contained in the file in order ot help give a complete understanding of how a perceptron works. 

Credit: Inspired and helped by Polycode (https://www.youtube.com/watch?v=kft1AJ9WVDk)


## Testing

In order to verify the success rate of the perceptron, multiple variables were chosen and the accuracy compared. 

Example:
training digits: 16 '3' digits and 16 'non-3' digits
training cycles: tested at 2,000 , 20,000 , and 200,000 cycles 
testing digits: 5 '3' digits and 5 'non-3' digits. Classification accuracy measured as percent accurate classifications. 

Accuracy Results:
2 cycles:       50% 
20 cycles:      50%
200 cycles:     60%
2000 cycles:    50%
20000 cycles:   60%
200K cycles:    60%
