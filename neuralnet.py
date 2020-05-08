import numpy as np

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
#num_samples = input_data.shape[0] #no. of rows in input_data = no. of samples/inputs

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

#start the training 
epochs = 0
max_epochs = 50000
while epochs < max_epochs:

    #go through the samples one-by-one
    for x, y in zip(input_data, output_data):

        #first (input) layer 
        #turn it into a column vector with l1 entries (x is a row vector with l1 entries)
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
        dW3 = np.dot(d3, a2.transpose()) 
        db3 = d3
        #second (hidden) layer
        d2 = np.dot(W3.transpose(), d3) * activation_function(z2, fn, True)
        dW2 = np.dot(d2, a1.transpose())
        db2 = d2

        #update weights and biases
        W2 = W2 - eta*dW2
        W3 = W3 - eta*dW3
        b2 = b2 - eta*db2
        b3 = b3 - eta*db3

    #output loss after every 10,000 epochs to see how we're doing 
    if epochs % 10000 == 0:
        print("Epoch {0}: loss {1}".format(epochs, np.mean(np.abs(nabla_cost(a3, y)))))
    
    epochs += 1