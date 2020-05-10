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
    
#go forward through the layers of the network from input to output 
def feedforward(a, z, num_layers, w, b, fn):
    for i in range(1, num_layers):
        z[i] = np.dot(w[i-1], a[i-1]) + b[i-1]
        a[i] = activation_function(z[i], fn, False)
    return a, z 

#backpropagation algorithm to calculate change in weights and biases to minimise cost function 
def backprop(a, z, delta, change_w, change_b, num_layers, w, b, y, fn):
    #first, calculate delta of output layer, delta[num_layers-1], using cost function 
    delta[num_layers-1] = nabla_cost(a[num_layers-1], y) * activation_function(z[num_layers-1], fn, True)
    #then, backpropagate from the output layer all the way to the input layer 
    for i in range(num_layers-2, -1, -1):
        change_w[i] += np.dot(delta[i+1], a[i].transpose())
        change_b[i] += delta[i+1]
        if i != 0: #no error for input layer 
            delta[i] = np.dot(w[i].transpose(), delta[i+1]) * activation_function(z[i], fn, True)
    return change_w, change_b

#update the weights and biases using stochastic gradient descent for each minibatch of size m 
def update(w, b, change_w, change_b, eta, m):
    for i in range(0, num_layers-1):
        w[i] = w[i] - (eta/m)*change_w[i]
        b[i] = b[i] - (eta/m)*change_b[i]
    return w, b 

#set learning rate
eta = 3

#predicting XOR of two bits 
#input 00, 01, 10, 11
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#corresponding output (XOR) 0, 1, 1, 0   
output_data = np.array([[0], [1], [1], [0]])

num_samples = input_data.shape[0] #no. of rows in input_data = no. of samples/inputs

#layers of the network 
np.random.seed(8)
layers = [2, 4, 2, 1] #number of neurons in each layer 
num_layers = len(layers) #number of layers in network 
w = [np.random.rand(layers[i+1], layers[i]) for i in range(0, len(layers)-1)] #initialising (num_layers-1) weight matrices 
b = [np.random.rand(layers[i+1], 1) for i in range(0, len(layers)-1)] #initialising (num_layers-1) bias vectors 

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

        #arrays to store changes in weights and biases for each minibatch, initialised to zero 
        change_w = [0 for i in range(0, num_layers-1)] #dw2 = change_w[0], dw3 = change_w[1]
        change_b = [0 for i in range(0, num_layers-1)] #db2= change_b[0], db3 = change_b[1]

        #go through the samples in each minibatch one-by-one
        for x, y in zip(minibatch_input_data, minibatch_output_data):

            z = [0 for i in range(0, num_layers)] #z2 = z[1], z3 = z[2]
            a = [0 for i in range(0, num_layers)] #a1 = a[0], a2 = a[1], a3 = a[2]
            delta = [0 for i in range(0, num_layers)] #d2 = delta[1], d3 = delta[2]
            
            #first (input) layer 
            #turn it into a column vector with layers[0](= n_i) entries (x is a row vector with n_i entries)
            a[0] = x.reshape(layers[0], 1)

            #feedforward to find output of each layer 
            feedforward(a, z, num_layers, w, b, fn)

            #backpropagate to calculate better weights and biases 
            backprop(a, z, delta, change_w, change_b, num_layers, w, b, y, fn)

        #update weights and biases after each minibatch
        update(w, b, change_w, change_b, eta, m)

    #output loss after every 10,000 epochs to see how we're doing 
    if epoch % 10000 == 0:
        print("Epoch {0}: loss {1}".format(epoch, np.mean(np.abs(nabla_cost(a[num_layers-1], y)))))
    
    epoch += 1

