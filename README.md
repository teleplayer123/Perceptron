# Machine Learning Algorithms and Me Learning... you are invited to learn as well!

## Starting in the brain!

For the first example, I want to share my attempt at writing a Perceptron, an algorithm invented in
1958 by Frank Rosenblatt. That's the extent of the history lesson portion. This is a good
place to start learning about the inner workings of machine learning.

### [Perceptron](perceptron.py)

The perceptron works like a human brain works. A neuron recieves an electrical signal from another neuron,
which only fires once the total strength of the signal reaches a certain threshold. Here neurons are represented
by numbers and are associated with weights which are also numbers. The neurons in a layer, represented as a list of 
floats, act as input into the next layer of neurons, which in this case is the output layer since this example has only
two layers. The weights start as random numbers, and get adjusted by calculating the error between output layer and 
our target layer. The perceptron will try to push the weights in the direction of the target in small steps, calculating
the error between the targets and the dot product of the inputs with the updated weights at each iteration. Due to dimensional
limitations of this algorithm, the perceptron can only learn the logic for "or" and "and" bit operations (i.e. 1|0==1, 1|1==1, 1&0==0, 1&1==1).
To get the perceptron to learn the logic for "xor" (i.e. 1|0==1, 1|1==0), a bias node is appended to each layer, based on
the input neurons. After many iterations, the perceptron learns the "xor" example with 100% accuracy. The epochs parameter is the
amount of iterations, and the learn_rate parameter controls how fast the perceptron learns. Finding the right balance between
these parameters is crucial to the success of the perceptrons ability to learn.
