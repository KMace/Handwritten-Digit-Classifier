import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

# Importing the MNIST hand-written digit data set 
(trainImages, trainLabels), (testImages, testLabels) = keras.datasets.mnist.load_data(path = "mnist.npz")

# Dividing by 255.0 in order to change pixel values from 0 - 255 to 0 - 1 as it ...
# makes the data easier to work with
trainImages = trainImages / 255.0
testImages = testImages / 255.0

# Creating the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)),
    keras.layers.Dense(156, activation = "relu"),
    keras.layers.Dense(10, activation = "softmax")
])

model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

# Training the model
model.fit(trainImages, trainLabels, epochs = 5)

# Testing the accuracy of the model
print()
testLoss, testAcc = model.evaluate(testImages, testLabels)
print("Overall Model Accuracy:", testAcc)

# Saving the model
model.save("digitNeuralNetwork.h5")

# Using the model to predict the entire set and output some of those predictions
predictions = model.predict(testImages)

for i in range(5):
    prediction = np.argmax(predictions[i])

    if str(prediction) == str(testLabels[i]):
        print("Correct prediction!")
    else:
        print("Incorrect prediction")

    plt.imshow(testImages[i], cmap = plt.cm.binary)
    plt.xlabel("Actual: " + str(testLabels[i]), fontsize = "large")
    plt.title("Prediction: " + str(prediction) + " with a confidence of: " + str(round(max(predictions[i]) * 100, 4)) + "%")
    plt.show()


# To open the saved model:
### model = keras.models.load_model("digitNeuralNetwork.h5")