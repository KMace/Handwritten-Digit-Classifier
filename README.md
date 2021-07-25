# Handwritten-Digit-Classifier
Neural network implemented in Python through 'Tensorflow' and 'Keras', with an accuracy of 97.5%.

Uses the MNIST digits classification dataset. 
Each of the digits are 28x28 pixels, so the input layer is 784 neurons (28 * 28). The hidden layer is 156 neurons, approximately 20% of the input layer (chosen arbitrarily), with a 'RELU' activation function. The output layer is of course 10 neurons in order to correspond to the possible digits 0 through 9, with a 'softmax' activation function.

Trained with 5 epochs as any more starts to give diminishing returns.
