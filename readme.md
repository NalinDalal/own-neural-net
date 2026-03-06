to do this today(06.03)


- [Neural Networks from Scratch — nnfs.io](https://nnfs.io/) (700-page guide)

| 3 | [Building your Deep NN: Step by Step](https://nbviewer.jupyter.org/github/amanchadha/coursera-deep-learning-specialization/blob/master/C1%20-%20Neural%20Networks%20and%20Deep%20Learning/Week%204/Building%20your%20Deep%20Neural%20Network%20-%20Step%20by%20Step/Building_your_Deep_Neural_Network_Step_by_Step_v8a.ipynb) | L-layer forward/backward, parameter init, cache — generalises Day 6's 2-layer net | Weight init strategies *Revisited Week 9 — DLS C2 W1 (zero vs random vs He)* |


# Neural Network from Scratch
neurons have some input, their corresponding weights and then you add some bias to it
the output is:
```python
inputs = [ 1 , 2 , 3 , 2.5 ]
weights = [[ 0.2 , 0.8 , -0.5 ,1],
[0.5,
- 0.91 , 0.26 ,
- 0.5 ],
[-0.26,- 0.27 , 0.17 , 0.87 ]]

biases = [ 2 , 3 , 0.5 ]
# Output of current layer
layer_outputs = []

# For each neuron
for neuron_weights, neuron_bias in zip (weights, biases):
    # Zeroed output of given neuron
    neuron_output = 0
    # For each input and weight to the neuron
    for n_input, weight in zip (inputs, neuron_weights):
        # Multiply this input by associated weight
        # and add to the neuron’s output variable
        neuron_output += n_input * weight
    # Add bias
    neuron_output += neuron_bias
    # Put neuron’s result to the layer’s output list
    layer_outputs.append(neuron_output)
print (layer_outputs)
```

then you can arrange multiple neurons on a sort of stack
so called neural network

fully connected neural network — every
neuron in the current layer has connections to every neuron from the previous layer

## Tensors, Arrays and Vectors
u know array, then you know about matrix, and 3d array
A tensor object is an object that can be represented as an array.

then u know about dot product of vector
```python
a = [ 1 , 2 , 3 ]
b = [ 2 , 3 , 4 ]
dot_product = a[ 0 ] * b[ 0 ] + a[ 1 ] * b[ 1 ] + a[ 2 ] * b[ 2 ]
print (dot_product)
```

use numpy to calculate the dot product
```python
import numpy as np
inputs = [ 1.0 , 2.0 , 3.0 , 2.5 ]
weights = [ 0.2 , 0.8 ,- 0.5 , 1.0 ]
bias = 2.0
outputs = np.dot(weights, inputs) + bias
print (outputs)
```

Layer of Neuron with NumPy
```python
import numpy as np
inputs = [ 1.0 , 2.0 , 3.0 , 2.5 ]
weights = [[ 0.2 , 0.8 ,- 0.5 , 1 ],
[ 0.5 ,- 0.91 , 0.26 ,- 0.5 ],
[ - 0.26 ,- 0.27 , 0.17 , 0.87 ]]
biases = [ 2.0 , 3.0 , 0.5 ]
layer_outputs = np.dot(weights, inputs) + biases
print (layer_outputs)

#array([ 4.8 1.21 2.385 ])
```

then u know how to find matrix multiplication

## Adding Layers
2/more hidden layer complicate the network
say Input layer with 4 features into a hidden layer with 3 neurons.
```python
import numpy as np
inputs = [[ 1 , 2 , 3 , 2.5 ], [ 2.0 , 5.0 ,-1.0,2.0],[ - 1.5 , 2.7 , 3.3 , -0.8 ]]
weights = [[ 0.2 , 0.8 ,- 0.5 , 1 ],
[ 0.5 ,- 0.91 , 0.26 ,- 0.5 ],
[ - 0.26 ,- 0.27 , 0.17 , 0.87 ]]
biases = [ 2 , 3 , 0.5 ]
weights2 = [[ 0.1 ,- 0.14 , 0.5 ],
[ - 0.5 , 0.12 ,- 0.33 ],
[ - 0.44 , 0.73 ,- 0.13 ]]
biases2 = [ - 1 , 2 ,- 0.5 ]


layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
print (layer2_outputs)
```
```
array([[ 0.5031 - 1.04185 - 2.03875 ],
[ 0.2434 - 2.7332 - 5.7633 ],
[ - 0.99314 1.41254 - 0.35655 ]])
```

creating a dataset, that is non-linear
```python
from nnfs.datasets import spiral_data
import numpy as np
import nnfs
import matplotlib.pyplot as plt

nnfs.init() #sets the random seed to 0 (by the default), creates a float32 dtype default, and overrides the original dot product from NumPy

X, y = spiral_data( samples = 100 , classes = 3 )   #create a dataset with as many classes as we want.
plt.scatter(X[:, 0 ],X[:, 1 ])
plt.show()

#adding colour to chart
plt.scatter(X[:, 0 ], X[:, 1 ], c = y, cmap = 'brg' )
plt.show()
```

## Dense Layer Class
```python
class Layer_Dense :
    def __init__ ( self , n_inputs , n_neurons ):
        # Initialize weights and biases(randomly)
        self.weights = 0.01 * np.random.randn(n_inputs,n_neurons) 
        self.biases = np.zeros((1 , n_neurons))
        pass # using pass statement as a placeholder

    # Forward pass
    def forward ( self , inputs ):
        # Calculate output values from inputs, weights and biases
        pass # using pass statement as a placeholder

# Create dataset
X, y = spiral_data( samples = 100 , classes = 3 )

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense( 2 , 3 )

# Perform a forward pass of our training data through this layer
dense1.forward(X)

# Let's see output of the first few samples:
print (dense1.output[: 5 ])
```

# Activation Functions
(chap-4)