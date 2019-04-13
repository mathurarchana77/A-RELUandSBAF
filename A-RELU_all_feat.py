from random import seed
from random import uniform
from random import randrange
from random import random
from csv import reader
from math import exp
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
import numpy as np
import csv
from decimal import Decimal
import matplotlib.pyplot as plt

 
# Load a CSV file
def loadCsv(filename):
        trainSet = []
        
        lines = csv.reader(open(filename, 'r'))
        dataset = list(lines)
        for i in range(len(dataset)):
                for j in range(4):
                        #print("DATA {}".format(dataset[i]))
                        dataset[i][j] = float(dataset[i][j])
                trainSet.append(dataset[i])
        return trainSet
 
# Convert string column to float
def str_column_to_float(dataset, column):
        for row in dataset:
                try:
                        row[column] = float(row[column])
                except ValueError:
                        print("Error with row",column,":",row[column])
                        pass
 
# Convert string column to integer
def str_column_to_int(dataset, column):
        class_values = [row[column] for row in dataset]
        unique = set(class_values)
        
        lookup = dict()
        for i, value in enumerate(unique):
                lookup[value] = i+1
        for row in dataset:
                row[column] = lookup[row[column]]
        print(lookup)
        return lookup
 
# Find the min and max values for each column
def dataset_minmax(dataset):
        minmax = list()
        stats = [[min(column), max(column)] for column in zip(*dataset)]
        return stats
 
# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
        for row in dataset:
                for i in range(len(row)-1):
                        row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
 
def splitData(dataset, sRatio):
        trainSet = []
        copy = list(dataset)
        trainSize = int(len(dataset) * sRatio)        
        seed(8)
        while len(trainSet) < trainSize:                
                index = randrange(len(copy))                
                trainSet.append(copy.pop(index))
        return [trainSet, copy]
 
# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
        correct = 0
        for i in range(len(actual)):
                if actual[i] == predicted[i]:
                        correct += 1
        return correct / float(len(actual)) * 100.0
 
# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(train_set, test_set, algorithm, *args):
                #print(test_set)
                predicted = algorithm(train_set, test_set, *args)
                #print(predicted)
                actual = [int(row[-1]) for row in test_set]
                #print(actual)
                accuracy = accuracy_metric(actual, predicted)
                cm = confusion_matrix(actual, predicted)
                print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in cm]))
                #confusionmatrix = np.matrix(cm)
                FP = cm.sum(axis=0) - np.diag(cm)
                FN = cm.sum(axis=1) - np.diag(cm)
                TP = np.diag(cm)
                TN = cm.sum() - (FP + FN + TP)
                print('False Positives\n {}'.format(FP))
                print('False Negetives\n {}'.format(FN))
                print('True Positives\n {}'.format(TP))
                print('True Negetives\n {}'.format(TN))
                TPR = TP/(TP+FN)
                print('Sensitivity \n {}'.format(TPR))
                TNR = TN/(TN+FP)
                print('Specificity \n {}'.format(TNR))
                Precision = TP/(TP+FP)
                print('Precision \n {}'.format(Precision))
                Recall = TP/(TP+FN)
                print('Recall \n {}'.format(Recall))
                Acc = (TP+TN)/(TP+TN+FP+FN)
                print('Áccuracy \n{}'.format(Acc))
                Fscore = 2*(Precision*Recall)/(Precision+Recall)
                print('FScore \n{}'.format(Fscore))
                k=cohen_kappa_score(actual, predicted)
                print('Çohen Kappa \n{}'.format(k))
                
        
 
# Calculate neuron activation for an input
def activate(weights, inputs):
        activation = weights[-1]
        for i in range(len(weights)-1):
                activation += weights[i] * inputs[i]
        return activation
 
# Transfer neuron activation
def transfer(act,k,n):
        if act <= 0:
                y=0
        else:
                y = k * pow(act,n)
        return y
 
# Forward propagate input to a network output
def forward_propagate(network, row):
        k=0.5
        n=1.1
        inputs = row
        for layer in network:
                new_inputs = []
                for neuron in layer:
                        activation = activate(neuron['weights'], inputs)                        
                        #print("weight {}".format(neuron['weights']))
                        #print("x {}".format(activation))
                        neuron['input'] = activation
                        neuron['output'] = transfer(activation,k,n)
                        #print(" y {}".format(neuron['output']))
                        new_inputs.append(neuron['output'])
                        #new_inputs.append(neuron['input'])
                inputs = new_inputs
        return inputs
 
# Calculate the derivative of an neuron output
def transfer_derivative(output,k,n):
        frac = 1/n
        p = 1-frac
        derv = pow(k,frac)*n
        derv = derv * pow(output,p)
        #print(derv)
        return derv
 
# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
        k=0.5
        n=1.1
        for i in reversed(range(len(network))):
                layer = network[i]
                errors = list()
                if i != len(network)-1:
                        for j in range(len(layer)):
                                error = 0.0
                                for neuron in network[i + 1]:
                                        error += (neuron['weights'][j] * neuron['delta'])
                                errors.append(error)
                else:
                        for j in range(len(layer)):
                                neuron = layer[j]
                                errors.append(expected[j] - neuron['output'])
                for j in range(len(layer)):
                        neuron = layer[j]
                        neuron['delta'] = errors[j] * transfer_derivative(neuron['output'], k,n)
 
# Update network weights with error
def update_weights(network, row, l_rate):
        for i in range(len(network)):
                inputs = row[:-1]
                
                if i != 0:
                        inputs = [neuron['output'] for neuron in network[i - 1]]
                for neuron in network[i]:
                        for j in range(len(inputs)):
                                temp = l_rate * neuron['delta'] * inputs[j] + mu * neuron['prev'][j]                                
                                neuron['weights'][j] += temp
                                #print("neuron weight{} \n".format(neuron['weights'][j]))
                                neuron['prev'][j] = temp
                        temp = l_rate * neuron['delta'] + mu * neuron['prev'][-1]
                        neuron['weights'][-1] += temp
                        neuron['prev'][-1] = temp
                #print("neuron {}".format(neuron['weights']))
                                
 
# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
        plotting = list()
        rang = list(range(1,n_epoch+1))
        for epoch in range(n_epoch):
                i = 1                
                msq = 0
                error = 0
                for row in train:
                        outputs = forward_propagate(network, row)
                        #print(outputs)
                        expected = [0 for i in range(n_outputs)]
                        expected[int(row[-1])] = 1

                        r = np.array(expected)
                        o = np.array(outputs)
                        r = r.astype(float)
                        o = o.astype(float)
                        #print("{}      {}".format(row_new,o))
                        err = np.subtract(r,o)
                        #print(err)
                        err = np.sum(err**2)/len(err)
                        #print(err)
                        error = error+err
                        i = i+1
                        #print("expected row{}\n".format(expected))
                        backward_propagate_error(network, expected)
                        update_weights(network, row, l_rate)
                        #print(network)
                mse = error/i
                plotting.append(mse)
        plt.plot(rang, plotting, 'ro')
        plt.xlabel('no of epochs - 1000')
        plt.ylabel('Mean square error')
        plt.show()
        print("mean square error{}\n".format(mse))
 
# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
        network = list()
        hidden_layer = [{'weights':[round(uniform(0,0.5),4) for i in range(n_inputs + 1)], 'prev':[0 for i in range(n_inputs+1)]} for i in range(n_hidden)]        
        network.append(hidden_layer)
        #hidden_layer = [{'weights':[random() for i in range(n_hidden + 1)], 'prev':[0 for i in range(n_hidden+1)]} for i in range(n_hidden)]
        #network.append(hidden_layer)
        output_layer = [{'weights':[round(uniform(0,0.5),4) for i in range(n_hidden + 1)],'prev':[0 for i in range(n_hidden+1)]} for i in range(n_outputs)]
        network.append(output_layer)
        #print(network)
        return network
 
# Make a prediction with a network
def predict(network, row):
        outputs = forward_propagate(network, row)
        #print(outputs)
        value = outputs.index(max(outputs))
        return value
 
# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(train, test, l_rate, n_epoch, n_hidden):
        n_inputs = len(train[0]) - 1
        n_outputs = len(set([row[-1] for row in train]))
        print("output {}".format(n_outputs))
        network = initialize_network(n_inputs, n_hidden, n_outputs)
        train_network(network, train, l_rate, n_epoch, n_outputs)
        #print("network {}\n".format(network))
        predictions = list()
        for row in test:
                prediction = predict(network, row)
                predictions.append(prediction)
        return(predictions)
 
# Test Backprop on Seeds dataset
seed(8)
# load and prepare data
filename = 'dataset-rocky-all-feats-run.csv'
dataset = loadCsv(filename)
for i in range(len(dataset[0])-1):
        str_column_to_float(dataset, i)
# convert class column to integers
#str_column_to_int(dataset, len(dataset[0])-1)
# normalize input variables

minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)
# evaluate algorithm
sRatio = 0.80
trainingSet, testSet = splitData(dataset, sRatio)
l_rate = 0.01
mu=0.001
print("learning rate, momentum",l_rate,mu)
n_epoch = 300
n_hidden = 12
evaluate_algorithm(trainingSet,testSet, back_propagation, l_rate, n_epoch, n_hidden)
