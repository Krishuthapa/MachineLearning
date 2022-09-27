#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 08:49:33 2022

@author: krishuthapa
"""

import numpy as np
import matplotlib.pyplot as plt

# Reading the training data csv file and assigning to variable.

training_data = np.genfromtxt('fashion-mnist_train.csv',delimiter=',', skip_header=1, filling_values=0)
testing_data = np.genfromtxt('fashion-mnist_test.csv', delimiter=',', skip_header=1, filling_values = 0)


#initial values setup

data_width = np.shape(training_data)[1] - 1


# Plain Perceptron Model
def getPerceptronModelWeights(max_itr, dataset = training_data):
    
    learning_rate = 1
    weights = np.zeros(data_width)

    for itr in range(max_itr):
        for data_point in dataset:
            label = data_point[0]
            x_values = data_point[1:]
            
            f_x = np.sum(np.multiply(x_values,weights));
            
            # Here we have set even values as 1 and odd as -1.
            label_class = 1 if (label == 0 or (label%2) == 0) else -1
            
            
            if((label_class*f_x) <= 0):
                added_value = learning_rate * label_class * x_values
                weights = np.add(weights, added_value)
    
    return weights;


def checkPlainPerceptronModel(calculated_weights, dataset):
    error = 0;
    for data_point in dataset:
        
        label = data_point[0]
        label_class = 1 if (label == 0 or (label%2) == 0) else -1
    
        x_values = data_point[1:]
        
        f_x = np.sum(np.multiply(x_values,calculated_weights))
        
        result = 1 if f_x > 0 else -1
        
        if(label_class != result):
            error=error+1
    
    return error


# Averaged Perceptron Model

def getAveragedPerceptronModelWeights(max_itr,dataset = training_data):
    
    learning_rate = 1

    weights = np.zeros(data_width)
    count =1    

    stored_weights = [weights]
    stored_weight_counts = [count]
    
    for itr in range(max_itr):
        
        for data_point in dataset:
            label = data_point[0]
            x_values = data_point[1:]
            
            f_x = np.sum(np.multiply(x_values,weights));
            
            # Here we have set even values as 1 and odd as -1.
            label_class = 1 if (label == 0 or (label%2) == 0) else -1
            
            
            if((label_class*f_x) <= 0):
                added_value = learning_rate * label_class * x_values
                weights = np.add(weights, added_value)
                
                stored_weights.append(weights)
                stored_weight_counts.append(count)
                
                count = 1
            else:
                count = count+1
    
    
    return  [stored_weight_counts, stored_weights]

def checkAveragedPerceptronModel(weight_counts, stored_weights, dataset):

    error = 0;
    
    # This is the sum of all (c,w) achieved from the training
    
    for count in range(np.shape(weight_counts)[0]):
        stored_weights[count] = weight_counts[count] * stored_weights[count]
    
    cumulated_weights = np.sum(stored_weights, axis = 0)
    
    
    # Calculating the value for the given dataset.
    
    for data_point in dataset:
        
        label = data_point[0]
        label_class = 1 if (label == 0 or (label%2) == 0) else -1
    
        x_values = data_point[1:]
        
        f_x = np.sum(np.multiply(x_values,cumulated_weights))
        
        result = 1 if f_x > 0 else -1
        
        if(label_class != result):
            error=error+1
            
    return error


# Passive Aggressive Model


def getPassiveAggressiveModelWeights(max_itr,dataset = training_data):
    weights = np.zeros(data_width)

    for itr in range(max_itr):
        for data_point in dataset:
            label = data_point[0]
            x_values = data_point[1:]
            
            f_x = np.sum(np.multiply(x_values,weights));
            
            # Here we have set even values as 1 and odd as -1.
            label_class = 1 if (label == 0 or (label%2) == 0) else -1
            
            
            if(np.absolute(label_class*f_x) <1):
                learning_rate = (1 - f_x * label_class)/(np.linalg.norm(x_values)**2)
                
                added_value = learning_rate * label_class * x_values
                weights = np.add(weights, added_value)
    
    return weights;

def checkPassiveAggressiveModel(calculated_weights, dataset):
    error = 0;
    for data_point in dataset:
        
        label = data_point[0]
        label_class = 1 if (label == 0 or (label%2) == 0) else -1
    
        x_values = data_point[1:]
        
        f_x = np.sum(np.multiply(x_values,calculated_weights));
        
        result = 1 if f_x > 0 else -1
        
        if(label_class != result):
            error=error+1
    
    return error


    

# Plot Questions

def plotQuestionOneCurve():
    training_itr = 1
    
    plot_data_itr = []
    plot_data_perceptron = []
    plot_data_pa = []
    
    while training_itr <= 51:
       updated_weights = getPerceptronModelWeights(training_itr)      
       perceptron_errors = checkPlainPerceptronModel(updated_weights,training_data)
       
       plot_data_perceptron.append(perceptron_errors)
       
       pa_updated_weights = getPassiveAggressiveModelWeights(training_itr)
       pa_errors = checkPassiveAggressiveModel(pa_updated_weights,training_data)
       
       plot_data_pa.append(pa_errors)
       
       plot_data_itr.append(training_itr)
       
       
       training_itr = training_itr + 3
    
    plt.plot(plot_data_itr,plot_data_perceptron,label = "Perceptron Training Error")
    plt.plot(plot_data_itr,plot_data_pa,label = "Passive Aggressive Training Error")
    plt.legend()
    plt.xlabel('Training Iterations')
    plt.ylabel('Training Errors')
    plt.title("Online Learning Curve")
    plt.show()
    
def plotQuestionTwoCurve():
    training_itr = 1
    
    plot_data_itr = []
    
    perceptron_training_acc = []
    perceptron_testing_acc = []
    
    pa_training_acc = []
    pa_testing_acc = []
    
    total_count_training = np.shape(training_data)[0]
    total_count_testing = np.shape(testing_data)[0]

    while training_itr <= 22:
        
       updated_weights = getPerceptronModelWeights(training_itr)      
       
       perc_train_errors = checkPlainPerceptronModel(updated_weights,training_data)
       perc_test_errors = checkPlainPerceptronModel(updated_weights,testing_data)
       
       
       perceptron_training_acc.append(100 - (perc_train_errors/total_count_training)*100)
       perceptron_testing_acc.append(100 - (perc_test_errors/total_count_testing)*100)
       
       pa_updated_weights = getPassiveAggressiveModelWeights(training_itr)
       
       pa_train_errors = checkPassiveAggressiveModel(pa_updated_weights,training_data)
       pa_test_errors = checkPassiveAggressiveModel(pa_updated_weights,testing_data)
       
       pa_training_acc.append(100 - (pa_train_errors/total_count_training)*100)
       pa_testing_acc.append(100 - (pa_test_errors/total_count_testing)*100)
         
       plot_data_itr.append(training_itr)
       
       training_itr = training_itr + 3
    
    plt.plot(plot_data_itr,perceptron_training_acc,label = "Perceptron Train")
    plt.plot(plot_data_itr,perceptron_testing_acc,label = "Perceptron Test")
    
    plt.plot(plot_data_itr,pa_training_acc,label = "PA Train")
    plt.plot(plot_data_itr,pa_testing_acc,label = "PA Test")
    
    plt.legend()
    plt.xlabel('Training Iterations')
    plt.ylabel('Accuracy')
    plt.title("Training and Testing Accuracy")
    plt.show()
    

def plotQuestionThreeCurve():
    training_itr = 1
    
    plot_data_itr = []
    
    perceptron_training_acc = []
    perceptron_testing_acc = []
    
    avg_perc_training_acc = []
    avg_perc_testing_acc = []
    
    total_count_training = np.shape(training_data)[0]
    total_count_testing = np.shape(testing_data)[0]

    while training_itr <= 22:
        
       updated_weights = getPerceptronModelWeights(training_itr)      
       
       perc_train_errors = checkPlainPerceptronModel(updated_weights,training_data)
       perc_test_errors = checkPlainPerceptronModel(updated_weights,testing_data)
       
       
       perceptron_training_acc.append(100 - (perc_train_errors/total_count_training)*100)
       perceptron_testing_acc.append(100 - (perc_test_errors/total_count_testing)*100)
       
       update_weights_count_collection = getAveragedPerceptronModelWeights(training_itr)
       
       avg_perc_train_errors = checkAveragedPerceptronModel(update_weights_count_collection[0],
                             update_weights_count_collection[1],training_data)
       avg_perc_test_errors = checkAveragedPerceptronModel(update_weights_count_collection[0],
                             update_weights_count_collection[1],testing_data)
       
       avg_perc_training_acc.append(100 - (avg_perc_train_errors/total_count_training)*100)
       avg_perc_testing_acc.append(100 - (avg_perc_test_errors/total_count_testing)*100)
         
       plot_data_itr.append(training_itr)
       
       training_itr = training_itr + 3
    
    plt.plot(plot_data_itr,perceptron_training_acc,label = "Perceptron Train ")
    plt.plot(plot_data_itr,perceptron_testing_acc,label = "Perceptron Test ")
    
    plt.plot(plot_data_itr,avg_perc_training_acc,label = "Avg Perceptron Train")
    plt.plot(plot_data_itr,avg_perc_testing_acc,label = "Avg Perceptron Test")
    
    plt.legend(bbox_to_anchor =(1.05, 0.5))
    plt.xlabel('Training Iterations')
    plt.ylabel('Accuracy')
    plt.title("Training and Testing Accuracy")
    plt.show()
    
def plotQuestionFourCurve():
    training_itr = 1
    training_count = 100
    
    plot_data_count = []
    
    perceptron_testing_acc = []
    
    avg_perc_testing_acc = []
    
    pa_testing_acc = []
    
    total_count_testing = np.shape(testing_data)[0]

    while training_itr <= 20:
        
       updated_weights = getPerceptronModelWeights(training_itr,training_data[:training_count])      
       
       perc_test_errors = checkPlainPerceptronModel(updated_weights,testing_data)
       perceptron_testing_acc.append(100 - (perc_test_errors/total_count_testing)*100)
       
       update_weights_count_collection = getAveragedPerceptronModelWeights(training_itr,training_data[:training_count])
       
       avg_perc_test_errors = checkAveragedPerceptronModel(update_weights_count_collection[0],
                             update_weights_count_collection[1],testing_data)
       avg_perc_testing_acc.append(100 - (avg_perc_test_errors/total_count_testing)*100)
       
       pa_updated_weights = getPassiveAggressiveModelWeights(training_itr,training_data[:training_count])
       pa_test_errors = checkPassiveAggressiveModel(pa_updated_weights,testing_data)
       
       pa_testing_acc.append(100 - (pa_test_errors/total_count_testing)*100)
         
       plot_data_count.append(training_count)
       
       training_itr += 1
       training_count += 100
    
    plt.plot(plot_data_count,perceptron_testing_acc,label = "Perceptron Test ")
    plt.plot(plot_data_count,avg_perc_testing_acc,label = "Avg Perceptron Test")
    plt.plot(plot_data_count,pa_testing_acc,label = "Passive Aggressive Test")
    
    plt.legend(bbox_to_anchor =(1.05, 0.5))
    plt.xlabel('Training Data Count')
    plt.ylabel('Test Accuracy')
    plt.title("General Learning Curve")
    plt.show()
    
    

def main():
    input_value = input("Enter the question number for seeing the corresponding plot (1-4):")
    
    if(int(input_value) == 1):
        plotQuestionOneCurve()
        
    elif(int(input_value) == 2):
        plotQuestionTwoCurve()
        
    elif(int(input_value) == 3):
        plotQuestionThreeCurve()
        
    elif(int(input_value) == 4):
        plotQuestionFourCurve()
        
    else:
        print("Invalid Entry.")
            

main()
               
    

