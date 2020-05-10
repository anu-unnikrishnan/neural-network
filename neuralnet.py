#stochastic gradient descent, minibatch size > 1

import numpy as np
import random

#activation function and its derivative
def activation_function(z, fn, prime):

    #sigmoid
    if fn == 1: 
        res = 1/(1 + np.exp(-z)) 
        deriv = res*(1-res) 

    #add other activation functions 

    if prime == False:
        return res #activation
    return deriv #derivative of activation
    
#calculate nabla_a C, the derivative of the cost function
def nabla_cost(a, y):
    return a-y #a should be output activation a^L

#set learning rate
eta = 1

#input 00, 01, 10, 11
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
num_samples = input_data.shape[0] #no. of rows in input_data = no. of samples/inputs

#corresponding output (XOR) 0, 1, 1, 0   
output_data = np.array([[0], [1], [1], [0]])

#number of neurons in each layer
n_i = 2 #input layer
n_h = 3 #hidden layer
n_o = 1 #output layer

#start off with random weights and biases
np.random.seed(1)
W2 = np.random.rand(n_h, n_i)
W3 = np.random.rand(n_o, n_h)
b2 = np.random.rand(n_h, 1)
b3 = np.random.rand(n_o, 1)

#choose activation function for all layers: 1 = sigmoid 
fn = 1

#set size of minibatches (best to be a power of 2)
m = 2

#start the training 
epoch = 0
max_epochs = 50000
while epoch < max_epochs:

    #split all samples into groups of m random samples (minibatches of size m)
    #first, shuffle input_data and output_data in same way
    shuffled_data = list(zip(input_data, output_data))
    random.shuffle(shuffled_data)
    input_data, output_data = zip(*shuffled_data)

    #then, choose m rows (m samples) randomly 
    for k in range(0, num_samples, m):

        minibatch_input_data = input_data[k:k+m]
        minibatch_output_data = output_data[k:k+m]

        #arrays to store changes in weights and biases for each minibatch 
        dW3, dW2, db3, db2 = [], [], [], []

        #go through the samples in each minibatch one-by-one
        for x, y in zip(minibatch_input_data, minibatch_output_data):

            #first (input) layer 
            #turn it into a column vector with n_i entries (x is a row vector with n_i entries)
            a1 = x.reshape(n_i, 1)

            #FEEDFORWARD
            z2 = np.dot(W2, a1) + b2
            a2 = activation_function(z2, fn, False)
            z3 = np.dot(W3, a2) + b3
            a3 = activation_function(z3, fn, False)

            #BACKPROPAGATION
            #calculate change in weights and biases
            #third (output) layer 
            d3 = nabla_cost(a3, y) * activation_function(z3, fn, True)
            dW3.append(np.dot(d3, a2.transpose()) )
            db3.append(d3)
            #second (hidden) layer
            d2 = np.dot(W3.transpose(), d3) * activation_function(z2, fn, True)
            dW2.append(np.dot(d2, a1.transpose()))
            db2.append(d2)

        #update weights and biases after each minibatch
        W2 = W2 - (eta/m)*sum(dW2)
        W3 = W3 - (eta/m)*sum(dW3)
        b2 = b2 - (eta/m)*sum(db2)
        b3 = b3 - (eta/m)*sum(db3)

    #output loss after every 10,000 epochs to see how we're doing 
    if epoch % 10000 == 0:
        print("Epoch {0}: loss {1}".format(epoch, np.mean(np.abs(nabla_cost(a3, y)))))
    
    epoch += 1