#####################################################################################################################
#   CS 6375.003 - Assignment 3, Neural Network Programming
#   This is a starter code in Python 3.6 for a 2-hidden-layer neural network.
#   You need to have numpy and pandas installed before running this code.
#   Below are the meaning of symbols:
#   train - training dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   train - test dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   h1 - number of neurons in the first hidden layer
#   h2 - number of neurons in the second hidden layer
#   X - vector of features for each instance
#   y - output for each instance
#   w01, delta01, X01 - weights, updates and outputs for connection from layer 0 (input) to layer 1 (first hidden)
#   w12, delata12, X12 - weights, updates and outputs for connection from layer 1 (first hidden) to layer 2 (second hidden)
#   w23, delta23, X23 - weights, updates and outputs for connection from layer 2 (second hidden) to layer 3 (output layer)
#
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it in the README file.
#
#####################################################################################################################


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import sys


class NeuralNet:
    def __init__(self, train, activation,header = True, h1 = 4, h2 = 2):

        np.random.seed(1)
        # train refers to the training dataset
        # test refers to the testing dataset
        # h1 and h2 represent the number of nodes in 1st and 2nd hidden layers

        raw_input = pd.read_csv(train, header=None)
        # TODO: Remember to implement the preprocess method
        train_dataset = self.preprocess(raw_input)
        ncols = len(train_dataset.columns)
        nrows = len(train_dataset.index)
        self.X = train_dataset.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols-1)
        self.y = train_dataset.iloc[:, (ncols-1)].values.reshape(nrows, 1)
        self.activation = activation
        self.rows = nrows
        #
        # Find number of input and output layers from the dataset
        #
        input_layer_size = len(self.X[0])
        if not isinstance(self.y[0], np.ndarray):
            output_layer_size = 1
        else:
            # output_layer_size = len(np.unique(self.y))
            output_layer_size = len(self.y[0])

        # assign random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.w01 = 2 * np.random.random((input_layer_size, h1)) - 1
        self.X01 = self.X
        self.delta01 = np.zeros((input_layer_size, h1))
        self.w12 = 2 * np.random.random((h1, h2)) - 1
        self.X12 = np.zeros((len(self.X), h1))
        self.delta12 = np.zeros((h1, h2))
        self.w23 = 2 * np.random.random((h2, output_layer_size)) - 1
        self.X23 = np.zeros((len(self.X), h2))
        self.delta23 = np.zeros((h2, output_layer_size))
        self.deltaOut = np.zeros((output_layer_size, 1))

    #
    # TODO: I have coded the sigmoid activation function, you need to do the same for tanh and ReLu
    #
    def __activation(self, x):
        activation = self.activation
        if activation == "sigmoid":
            out1 = self.__sigmoid(x)
        elif activation == "tanh":
            out1 = self.__tanh(x)
        elif activation == "ReLu":
            out1 = self.__relu(x)

        return out1

    # sigmoid function
    def __sigmoid(self, x):
        return 1. / (1 + np.exp(-x))

    # tanh function
    def __tanh(self, x):
    	return np.tanh(x)

    # ReLu function
    def __relu(self, x):
    	# return np.log(1/(1+np.exp(x)))
        return np.maximum(0,x)

    #
    # TODO: Define the function for tanh, ReLu and their derivatives
    #
    def __activation_derivative(self, x):
        activation = self.activation
        if activation == "sigmoid":
            self.__sigmoid_derivative(x)
        elif activation == "tanh":
        	self.__tanh_derivative(x)
        elif activation == "ReLu":
        	self.__relu_derivative(x)

    # derivative of sigmoid function, indicates confidence about existing weight
    def __sigmoid_derivative(self, x):

        return x * (1. - x)

    # derivative of tanh function
    def __tanh_derivative(self, x):
    	return 1. - x * x

    # derivative of ReLu function  
    def __relu_derivative(self, x):
        return 1. * (x>0)

    #
    # TODO: Write code for pre-processing the dataset, which would include standardization, normalization,
    #   categorical to numerical, etc
    # Using Label Encoding to transform categorical to numerical
    # NA data is represented by the current column mean
    # Numerical data is transformed by its mean and sd

    def preprocess(self, data):
        num = data.shape[0]
        
        for column in data:
            if data[column].dtype == "object":
                data[column] = data[column].astype('category').cat.codes
            else:
                temp = data.iloc[: num, column].fillna(data[column]. mean())
                mean = temp.mean()
                std = temp.std()
                data[column] = (data[column] - mean) / std

        return data


    # Below is the training function
    def train(self, max_iterations = 1000, learning_rate = 0.000001):
        for iteration in range(max_iterations):
            out = self.forward_pass(self.X)           
            error = 0.5 * np.power((out - self.y), 2)
            self.backward_pass(out)
            update_layer2 = learning_rate * self.X23.T.dot(self.deltaOut)
            update_layer1 = learning_rate * self.X12.T.dot(self.delta23)
            update_input = learning_rate * self.X01.T.dot(self.delta12)

            self.w23 = self.w23 + update_layer2
            self.w12 = self.w12 + update_layer1
            self.w01 = self.w01 + update_input

        print("After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error)/ self.rows))
        print("The final weight vectors are (starting from input to output layers)")
        print("The first layer weight vectors: ")
        print(self.w01)
        print("The second layer weight vectors: ")
        print(self.w12)
        print("The third layer weight vectors: ")
        print(self.w23)


    def forward_pass(self, X):
        # pass our inputs through our neural network
        in1 = np.dot(X, self.w01)   
        self.X12 = self.__activation(in1)
        in2 = np.dot(self.X12, self.w12)
        self.X23 = self.__activation(in2)
        in3 = np.dot(self.X23, self.w23)
        out = self.__activation(in3)
        
        return out

    def backward_pass(self, out):
        # pass our inputs through our neural network
        self.compute_output_delta(out, self.activation)
        self.compute_hidden_layer2_delta(self.activation)
        self.compute_hidden_layer1_delta(self.activation)

    # TODO: Implement other activation functions
    def compute_output_delta(self, out, activation):
        if activation == "sigmoid":
            delta_output = (self.y - out) * (self.__sigmoid_derivative(out))
        elif activation == "tanh":
        	delta_output = (self.y - out) * (self.__tanh_derivative(out))
        elif activation == "ReLu":
            delta_output = (self.y - out) * (self.__relu_derivative(out))

        self.deltaOut = delta_output

    # TODO: Implement other activation functions
    def compute_hidden_layer2_delta(self, activation):
        if activation == "sigmoid":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__sigmoid_derivative(self.X23))
        elif activation == "tanh":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__tanh_derivative(self.X23))
        elif activation == "ReLu":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__relu_derivative(self.X23))   

        self.delta23 = delta_hidden_layer2

    # TODO: Implement other activation functions

    def compute_hidden_layer1_delta(self, activation):
        if activation == "sigmoid":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__sigmoid_derivative(self.X12))
        elif activation == "tanh":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__tanh_derivative(self.X12))
        elif activation == "ReLu":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__relu_derivative(self.X12))

        self.delta12 = delta_hidden_layer1

    # TODO: Implement other activation functions

    def compute_input_layer_delta(self, activation):
        if activation == "sigmoid":
            delta_input_layer = np.multiply(self.__sigmoid_derivative(self.X01), self.delta01.dot(self.w01.T))
        elif activation == "tanh":
            delta_input_layer = np.multiply(self.__tanh_derivative(self.X01), self.delta01.dot(self.w01.T))
        elif activation == "ReLu":
            delta_input_layer = np.multiply(self.__relu_derivative(self.X01), self.delta01.dot(self.w01.T))

        self.delta01 = delta_input_layer

    # TODO: Implement the predict function for applying the trained model on the  test dataset.
    # You can assume that the test dataset has the same format as the training dataset
    # You have to output the test error from this function

    def predict(self, test, header = True):

    	raw_test_input = pd.read_csv(test, header=None)
        # TODO: Remember to implement the preprocess method
        test_dataset = self.preprocess(raw_test_input)
        test_ncols = len(test_dataset.columns)
        test_nrows = len(test_dataset.index)

        test_Y = test_dataset.iloc[:, (test_ncols-1)].values.reshape(test_nrows, 1)       
        test_X = test_dataset.iloc[:, 0:(test_ncols-1)].values.reshape(test_nrows, test_ncols-1)

        test_out = self.forward_pass(test_X)

       
        test_error = 0.5 * np.power((test_Y - test_out), 2)


        return np.sum(test_error) / test_nrows


if __name__ == "__main__":
    neural_network = NeuralNet("adultTrain.csv", sys.argv[1])
    neural_network.train()
    testError = neural_network.predict("adultTest.csv")
    print("The testError is")
    print(testError)

