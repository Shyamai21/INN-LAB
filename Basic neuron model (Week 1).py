#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

class Neuron:
    def __init__(self, num_inputs):
        # Initialize weights and bias with random values
        self.weights = np.random.rand(num_inputs)
        self.bias = np.random.rand()
        
    def sigmoid(self, x):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-x))
    
    def forward(self, inputs):
        # Calculate the weighted sum of inputs, add bias, and apply activation function
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        output = self.sigmoid(weighted_sum)
        return output

# Example usage:
num_inputs = 3
neuron = Neuron(num_inputs)

# Generate some random input values for testing
inputs = np.random.rand(num_inputs)

# Get the output from the neuron
output = neuron.forward(inputs)

# Print the results
print("Inputs:", inputs)
print("Weights:", neuron.weights)
print("Bias:", neuron.bias)
print("Output:", output)


# In[ ]:




