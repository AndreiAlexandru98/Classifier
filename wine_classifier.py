#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skeleton code for CW2 submission. 
We suggest you implement your code in the provided functions
Make sure to use our print_features() and print_predictions() functions
to print your results
"""

from __future__ import print_function
import math
import operator
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from utilities import load_data, print_features, print_predictions

CLASS_1_C = r'#3366ff'
CLASS_2_C = r'#cc3300'
CLASS_3_C = r'#ffc34d'

MODES = ['feature_sel', 'knn', 'alt', 'knn_3d', 'knn_pca']    

####----------------------------------------------------------------------------------------------------- 
#### MODEL: Feature selection 
def feature_selection(train_set, train_labels, **kwargs):
    n_features = train_set.shape[1]
    fig, ax = plt.subplots(n_features, n_features)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.2, hspace=0.4)

    colours = np.zeros_like(train_labels, dtype = np.object)
    colours[train_labels == 1] = CLASS_1_C
    colours[train_labels == 2] = CLASS_2_C
    colours[train_labels == 3] = CLASS_3_C

    for i in range (0, 13) :
        for j in range (0, 13) :
            ax[i,j].scatter(train_set[:, i], train_set[:, j], c = colours)
            ax[i,j].set_title('{} vs {}'.format(i, j), pad= -1)
    
    ####Uncomment to show the features matrix
    #plt.show()
    return [10,12]

####-----------------------------------------------------------------------------------------------------
#### Reduce data
def reduce_data(train_set, test_set, selected_features):
    if( selected_features != 0):
        train_set_red = train_set[:, selected_features]
        test_set_red = test_set[:, selected_features]
        return train_set_red, test_set_red
    else:
        return train_set, test_set


#### Distance calculator (euclidean)
def distanceCalculator(testInstance, trainInstance, length):
    distance = 0
    for x in range(length):
        distance += np.square(testInstance[x] - trainInstance[x])
    
    return np.sqrt(distance)


#### Get neighbors
def getNeighbours(train_set, testInstance, k):
    distances = {}
    length = len(testInstance)
    
    #### Calculate distances
    for x in range(len(train_set)):
        dist = distanceCalculator(testInstance, train_set[x], length)
        distances[x] = dist
        
    
    #### Sort distances
    sorted_d = sorted(distances.items(), key=operator.itemgetter(1))
    
    neighbors = []
    #### Store closest neighbors
    for x in range(k):
        neighbors.append(sorted_d[x][0])
        
    return neighbors
    
    
#### Calculate predictions
def calculatePredictions(train_set, testInstance, k, train_labels):
    votes = {}
    neighbors = getNeighbours(train_set, testInstance, k)
    #### Count votes
    for x in range(len(neighbors)):
        response = train_labels[neighbors[x]]
 
        if response in votes:
            votes[response] = votes[response] + 1
        else:
            votes[response] = 1

    #### Sort votes by the biggest 
    sortedVotes = sorted(votes.items(), key=operator.itemgetter(1), reverse=True)
    return(sortedVotes[0][0])


#### MODEL: K-Nearest Neighbour
def knn(train_set, train_labels, test_set,test_labels, selected_values, k, **kwargs):
    predictions=[]
    #### Reduce data
    train_set_red, test_set_red = reduce_data(train_set, test_set, selected_values)
    #### Get predictions
    for x in range(len(test_set_red)):
       result = calculatePredictions(train_set_red, test_set_red[x], k, train_labels)
       predictions.append(result)
    
    #### Calculate accuracy
    accurate = 0
    for x in range(len(test_labels)):
       if(test_labels[x] == predictions[x]):
           accurate = accurate + 1
    accuracy =  (accurate * 100 / len(test_labels))
    #### Uncomment to print accuracy
    #print(accuracy)
    
    confusion = confusion_matrix(test_labels, predictions)
    #### Uncomment to print confusion matrix 
    #print(confusion)
    plt.matshow(confusion)
    plt.title('Confusion matrix of the classifier')
    plt.colorbar()
    #### Uncomment to plot confusion matrix 
    #plt.show()
    
    return predictions


####-----------------------------------------------------------------------------------------------------
#### Separate data by their labels
def separateData(train_set, train_labels):
    separated = {}
    for i in range(len(train_set)):
        label = train_labels[i]
        data  = train_set[i]
        if( label not in separated):
             separated[label] = []
        separated[label].append(data)

    return separated
 
 
#### Mean calculator
def mean(n):
    return sum(n)/float(len(n))
 
 
#### Standard deviation calculator
def stdev(n):
    variance = sum([pow(x-mean(n),2) for x in n])/float(len(n)-1)
    return math.sqrt(variance)


#### Summarize data
def summarize(train_set, train_labels):
    #### Separate data
    separated = separateData(train_set, train_labels)
    summaries = {}
    #### Summarize data by label
    for label, instances in separated.items():      
        summaries[label] = [(mean(attribute), stdev(attribute)) for attribute in zip(*instances)]
    
    return summaries


#### Calculate probability
def calculate(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent


#### Get probabilities
def getProbabilities(summaries, testInstance):
    probabilities = {}
    for label, labelSummaries in summaries.items():
        probabilities[label] = 1
        for i in range(len(labelSummaries)):
            mean, stdev = labelSummaries[i]
            x = testInstance[i]
            probabilities[label] = probabilities[label] * calculate(x, mean, stdev)
    return probabilities


#### Predict best label
def predict(summaries, testInstance):
	probabilities = getProbabilities(summaries, testInstance)
	bestLabel, bestProbability = None, -1
	for label, probability in probabilities.items():
		if bestLabel is None or probability > bestProbability:
			bestProbability = probability
			bestLabel = label
	return bestLabel


#### Get predictions
def getPredictions(summaries, test_set):
    predictions = []
    for i in range(len(test_set)):
        result = predict(summaries, test_set[i])
        predictions.append(result)
    return predictions


#### MODEL: Alternative( Naive Bayes classifier)
def alternative_classifier(train_set, train_labels, test_set, selected_values, **kwargs):
    train_set_red, test_set_red = reduce_data(train_set, test_set, selected_values)
    #### Get summaries
    summaries = summarize(train_set_red, train_labels)
    #### Get predictions
    predictions = getPredictions(summaries, test_set_red)
    
    accurate = 0
    for x in range(len(test_labels)):
       if(test_labels[x] == predictions[x]):
           accurate = accurate + 1
    accuracy =  (accurate * 100 / len(test_labels))
    #### Uncomment to print accuracy
    #print(accuracy)
    
    return predictions


####-----------------------------------------------------------------------------------------------------
#### MODEL: K-Nearest Neighbour with 3 features
def knn_three_features(train_set, train_labels, test_set, k, **kwargs):
    predictions = knn(train_set, train_labels, test_set, test_labels, [10, 12, 11], k)
    
    return predictions


####-----------------------------------------------------------------------------------------------------
#### MODEL: K-Nearest Neighbour with PCA
def knn_pca(train_set, train_labels, test_set, k, n_components=2, **kwargs):
    #### PCA fit and transformation
    pca = PCA(n_components)
    pca.fit(train_set)
    pca_train_set = pca.transform(train_set)
    pca_test_set = pca.transform(test_set)
    
    #### Plotting the data
    n_features = train_set.shape[1]
    fig, ax = plt.subplots(2, 2)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.2, hspace=0.4)

    colours = np.zeros_like(train_labels, dtype = np.object)
    colours[train_labels == 1] = CLASS_1_C
    colours[train_labels == 2] = CLASS_2_C
    colours[train_labels == 3] = CLASS_3_C

    for i in range (0, 2) :
        for j in range (0, 2) :
            ax[i,j].scatter(pca_train_set[:, i], pca_train_set[:, j], c = colours)
    
    ####Uncomment to show the features matrix
    #plt.show()
    
    #### Get predictions
    predictions = knn(pca_train_set, train_labels, pca_test_set, test_labels, 0, k)
    
    return predictions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs=1, type=str, help='Running mode. Must be one of the following modes: {}'.format(MODES))
    parser.add_argument('--k', nargs='?', type=int, default=1, help='Number of neighbours for knn')
    parser.add_argument('--train_set_path', nargs='?', type=str, default='data/wine_train.csv', help='Path to the training set csv')
    parser.add_argument('--train_labels_path', nargs='?', type=str, default='data/wine_train_labels.csv', help='Path to training labels')
    parser.add_argument('--test_set_path', nargs='?', type=str, default='data/wine_test.csv', help='Path to the test set csv')
    parser.add_argument('--test_labels_path', nargs='?', type=str, default='data/wine_test_labels.csv', help='Path to the test labels csv')
    
    args = parser.parse_args()
    mode = args.mode[0]
    
    return args, mode


if __name__ == '__main__':
    args, mode = parse_args() # get argument from the command line
    
    # load the data
    train_set, train_labels, test_set, test_labels = load_data(train_set_path=args.train_set_path, 
                                                                       train_labels_path=args.train_labels_path,
                                                                       test_set_path=args.test_set_path,
                                                                       test_labels_path=args.test_labels_path)
    if mode == 'feature_sel':
        selected_features = feature_selection(train_set, train_labels)
        print_features(selected_features)
    elif mode == 'knn':
        predictions = knn(train_set, train_labels, test_set, test_labels, [10, 12], args.k)
        print_predictions(predictions)
    elif mode == 'alt':
        predictions = alternative_classifier(train_set, train_labels, test_set, [10, 12])
        print_predictions(predictions)
    elif mode == 'knn_3d':
        predictions = knn_three_features(train_set, train_labels, test_set, args.k)
        print_predictions(predictions)
    elif mode == 'knn_pca':
        prediction = knn_pca(train_set, train_labels, test_set, args.k)
        print_predictions(prediction)
    else:
        raise Exception('Unrecognised mode: {}. Possible modes are: {}'.format(mode, MODES))