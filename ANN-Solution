# The average root mean square error (RMSE) of the model is  0.39
#Importing the important modules
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
import tensorflow.keras as keras

from keras.models import Sequential
from keras.layers import Dense

#Loading the data
dataset = load_boston()

#Scaling all the features so that they are inbetween 0 and 1
x_scaled = scale(dataset.data)
y_scaled = scale(dataset.target)

#Dividing the data between training and testing
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled)

#Neural Network:
#Sequential: A linearly connected layers

#Activation: Defines the output of that node given an input or set of inputs. ReLU is a popular one

#Input layer: Takes in the 13 features of each point in the datset and passes it though an activation method

#Hidden Layer: Outputs 13 points, after passing through an activation method

#Output Layer: Outputs one point (the guess), after using the previous layer's points and passing it through an activation method. This time, it's linear because we are using linear regression
model = Sequential([
   Dense(20, activation="relu", input_dim=13),
   Dense(13, activation="relu"),
   Dense(1, activation="linear")
])

#Compiler:
#Loss: sets our metric of error to MSE

#Optimizer: Tries to decrease the metric of error over time. ADAM is a popular open

#Metrics: The "scores" that will be given after the learning is finished
model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error", "mean_absolute_error"])

#Fitting:
#Epochs: The amount of times the neural network trains.

#Validation Split: An extra testing set based off the training set for the neural network to compare results with.
model.fit(x_train, y_train, epochs=1000, validation_split=0.2)


#Evaluating how good our model was
loss, mse, mae = model.evaluate(x_test, y_test)
print('The average root mean square error (RMSE) of the model is {:5.2f}'.format(np.sqrt(mse)))
