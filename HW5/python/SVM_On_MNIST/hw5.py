#! /usr/bin/env python3
'''
    Author      : BCC
    Date        : 2022/05/02
'''

import argparse
import math
import sys
import re
import copy
import numpy as np
import matplotlib.pyplot as plt
import csv
from csv import reader
from libsvm.svmutil import *
from scipy.spatial import distance
from matplotlib.pyplot import MultipleLocator #for setting of scale of separating along with x-axis & y-axis.

#########################
#    Class-Definition   #
#########################

class color:
   PURPLE = '\033[1;35;48m'
   CYAN = '\033[1;36;48m'
   BOLD = '\033[1;37;48m'
   BLUE = '\033[1;34;48m'
   GREEN = '\033[1;32;48m'
   YELLOW = '\033[1;33;48m'
   RED = '\033[1;31;48m'
   BLACK = '\033[1;30;48m'
   UNDERLINE = '\033[4;37;48m'
   END = '\033[1;37;0m'

#########################
#     Main-Routine      #
#########################
def main():
    #Process the argument
    #print(f"> ArgumentParser...")
    (input_train_x, input_train_y, input_test_x, input_test_y, kfold, search_lin, search_pol, search_rbf, task, is_debug) = ArgumentParser()

    #Get the input training/test data points
    #print(f"> ReadInputfile...")
    train_x = ReadInputFile(input_train_x)
    train_y = ReadInputFile(input_train_y)
    test_x  = ReadInputFile(input_test_x)
    test_y  = ReadInputFile(input_test_y)

    #Perform Task 1 - Use different kernel function(linear, polynomial and RBF) to compare the performance.
    if(task == 1):
        SVMModelComparison(train_x, train_y, test_x, test_y)

    #Perform Task 2 - Grid search for linear, polynomial and RBF kernel for C-SVC
    if(task == 2):
        GridSearchOpt(train_x, train_y, test_x, test_y, kfold, search_lin, search_pol, search_rbf, is_debug)

    #Perform Task 3 - Use Self-Defined New Kernel: linear+RBF
    if(task == 3):
        SelfDefinedKernelSVM(train_x, train_y, test_x, test_y, kfold, is_debug)

    if(is_debug):
        '''
        print(f"train_x.shape = {train_x.shape}")
        print(f"train_y.shape = {train_y.shape}")
        print(f"test_x.shape = {test_x.shape}")
        print(f"test_y.shape = {test_y.shape}")

        for images in input_train_y:
            for index, pixel in enumerate(images):
                print(f"{int(pixel)}")

        for images in input_train_x:
            for index, pixel in enumerate(images):
                if(index == 28*28-1):
                    if(pixel==0):
                        print(f"0")
                    elif(pixel==1):
                        print(f"1")
                    else:
                        print(f"{pixel}")
                else:
                    if(pixel==0):
                        print(f"0,", end='')
                    elif(pixel==1):
                        print(f"1,", end='')
                    else:
                        print(f"{pixel},", end='')

        for images in input_test_y:
            for index, pixel in enumerate(images):
                print(f"{int(pixel)}")

        for images in input_test_x:
            for index, pixel in enumerate(images):
                if(index == 28*28-1):
                    if(pixel==0):
                        print(f"0")
                    elif(pixel==1):
                        print(f"1")
                    else:
                        print(f"{pixel}")
                else:
                    if(pixel==0):
                        print(f"0,", end='')
                    elif(pixel==1):
                        print(f"1,", end='')
                    else:
                        print(f"{pixel},", end='')
        '''

#########################
#     Sub-Routine       #
#########################
def ArgumentParser():
    input_train_x     = None
    input_train_y     = None
    input_test_x      = None
    input_test_y      = None
    kfold             = 5
    search_lin        = 0
    search_pol        = 0
    search_rbf        = 0
    task              = 0
    is_debug          = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_train_x", "-input_train_x",     help="The input file name of the input training data points x.")
    parser.add_argument("--input_train_y", "-input_train_y",     help="The input file name of the input training data points y.")
    parser.add_argument("--input_test_x",  "-input_test_x",      help="The input file name of the input testing data points x.")
    parser.add_argument("--input_test_y",  "-input_test_y",      help="The input file name of the input testing data points y.")
    parser.add_argument("--kfold",         "-kfold",             help="The number of the k-fold cross validation for tuning hyper parameters in task 2. Default is 5.")
    parser.add_argument("--search_lin",    "-search_lin",        help="Set 1 to do grid search for linear kernel. 0, otherwise. Default is 0.")
    parser.add_argument("--search_pol",    "-search_pol",        help="Set 1 to do grid search for polynomial kernel. 0, otherwise. Default is 0.")
    parser.add_argument("--search_rbf",    "-search_rbf",        help="Set 1 to do grid search for rbf kernel. 0, otherwise. Default is 0.")
    parser.add_argument("--task",          "-task",              help="Set 1 to perform task1. Set 2 to perform task 2. Set 3 to perform task3. Default is 1.")
    parser.add_argument("--is_debug",      "-isd",               help="1 for debug mode; 0 for normal mode.")

    args = parser.parse_args()

    if(args.input_train_x):
        input_train_x  = args.input_train_x
    if(args.input_train_y):
        input_train_y  = args.input_train_y
    if(args.input_test_x):
        input_test_x   = args.input_test_x
    if(args.input_test_y):
        input_test_y   = args.input_test_y
    if(args.kfold):
        kfold          = int(args.kfold)
    if(args.search_lin):
        search_lin     = int(args.search_lin)
    if(args.search_pol):
        search_pol     = int(args.search_pol)
    if(args.search_rbf):
        search_rbf     = int(args.search_rbf)
    if(args.task):
        task           = int(args.task)
    if(args.is_debug):
        is_debug       = int(args.is_debug)

    if(input_train_x == None):
        print(f"Error: You should set '--input_train_x' or '-input_train_x' for the input of training data points x.")
    if(input_train_y == None):
        print(f"Error: You should set '--input_train_y' or '-input_train_y' for the input of training data points x.")
    if(input_test_x == None):
        print(f"Error: You should set '--input_test_x' or '-input_test_x' for the input of testing data points x.")
    if(input_test_y == None):
        print(f"Error: You should set '--input_test_y' or '-input_test_y' for the input of testing data points x.")
        sys.exit()

    if(is_debug):
        print(f"input_train_x   = {input_train_x}")
        print(f"input_train_y   = {input_train_y}")
        print(f"input_test_x    = {input_test_x}")
        print(f"input_test_y    = {input_test_y}")
        print(f"kfold           = {kfold}")
        print(f"search_lin      = {search_lin}")
        print(f"search_pol      = {search_pol}")
        print(f"search_rbf      = {search_rbf}")
        print(f"task            = {task}")
        print(f"is_debug        = {is_debug}")

    return (input_train_x, input_train_y, input_test_x, input_test_y, kfold, search_lin, search_pol, search_rbf, task, is_debug)

def ReadInputFile(input_file):
    input_data = []
    with open(input_file, 'r') as read_obj:
        csv_reader = reader(read_obj)
        # Iterate over each row in the csv using reader object
        for row in csv_reader:
            row_data = [float(element) for element in row]
            input_data.append(row_data)

    return np.array(input_data)

def BuildLinearKernel(phi_x1, phi_x2):
    #phi_x1: 5000x784 or 2500x784, phi_x2: train, 5000x785
    linear_kernel = phi_x1@(phi_x2.T) #5000x5000 or 5000x2500

    return linear_kernel

def BuildRBFKernel(phi_x1, phi_x2, gamma):
    #phi_x1: 5000x784 or 2500x784, phi_x2: train, 5000x785
    row = phi_x1.shape[0]
    col = phi_x2.shape[0]
    kernel_mat = []

    dist_matrix = distance.cdist(phi_x1, phi_x2, 'sqeuclidean')

    for i in range(row):
        row_arr = []
        for j in range(col):
            kernel_val = np.exp(-1*gamma*dist_matrix[i][j])
            row_arr.append(kernel_val)
        kernel_mat.append(row_arr)
    return np.array(kernel_mat)

def AugmentIDColumn(input_matrix):
    rows = len(input_matrix)
    cols = len(input_matrix[0])
    result_matrix = []
    id_num        = 1

    for i in range(rows):
        row = []
        for j in range(cols+1):
            if(j==0):
                row.append(id_num)
            else:
                row.append(input_matrix[i][j-1])
        id_num += 1
        result_matrix.append(row)
    return np.array(result_matrix)

def BuildMixedKernel(phi_x1, phi_x2, gamma):
    #phi_x1: 5000x784 or 2500x784, phi_x2: train, 5000x785
    linear_kernel = BuildLinearKernel(phi_x1, phi_x2)
    rbf_kernel    = BuildRBFKernel(phi_x1, phi_x2, gamma)
    mixed_kernel  = linear_kernel + rbf_kernel

    #LIBSVM needs to add ID column at the front
    result_kernel = AugmentIDColumn(mixed_kernel)

    return result_kernel

def SelfDefinedKernelSVM(train_x, train_y, test_x, test_y, kfold, is_debug):
    train_y = train_y.ravel()
    test_y  = test_y.ravel()

    #Do the grid search for the hyper parameters: C and gamma
    print(f"")
    print(f"")
    print(f"<------------ Grid Search & Test for Sell-Defined Kernel: Linear + RBF------------>")
    svm_type_def = '-s 0 -t 4'
    param        = svm_parameter(svm_type_def)
    print(f"svm_type = {param.svm_type}, kernel_type = {param.kernel_type}, kfold = {kfold}")

    #grid search
    best_acc = -1
    best_c   = -1
    best_g   = -1
    sweep_c  = []
    sweep_g  = [float(1/784)]
    for i in range(-13, 14):
        sweep_c.append(np.power(2.0, i))
    for i in range(-13, 14):
        sweep_g.append(np.power(2.0, i))

    for c_value in sweep_c:
        for g_value in sweep_g:
            #Build Mixed Kernel: Linear+RBF
            result_kernel = BuildMixedKernel(train_x, train_x, g_value)

            #Setting parameters for svm
            param = svm_parameter(svm_type_def+' -c '+str(c_value)+' -g '+str(g_value)+' -v '+str(kfold)+' -q')
            prob  = svm_problem(train_y, result_kernel, isKernel=True) #Need to set isKernel=True when using precomputed kernel type

            if(is_debug):
                print(f"param.svm_type = {param.svm_type}, param.kernel_type = {param.kernel_type}, param.C = {param.C}, param.gamma = {param.gamma}, kfold = {kfold}")

            accuracy = svm_train(prob, param)

            if(accuracy > best_acc):
                best_acc = accuracy
                best_c   = c_value
                best_g   = g_value
                print(f"Currently best C = {best_c}, best gamma = {best_g}, accuracy = {best_acc}%")

    print(f"> --------------------------------------------------------------")
    print(f"> Done for grid search: ")
    print(f"Best hyper parameter C = {best_c}, gamma = {best_g}, with highest accuracy = {best_acc}%")
    print(f"> --------------------------------------------------------------")

    print(f"> Testing with test datasets: ")

    #testing
    result_train_kernel = BuildMixedKernel(train_x, train_x, best_g)
    prob                = svm_problem(train_y, result_train_kernel, isKernel=True) #retrain the model with opt hyper parameters
    param               = svm_parameter(svm_type_def+' -c '+str(best_c)+' -g '+str(best_g)+' -q')
    model               = svm_train(prob, param)
    result_test_kernel  = BuildMixedKernel(test_x, train_x, best_g)
    svm_predict(test_y, result_test_kernel, model)

def GridSearchKernelLinear(train_x, train_y, test_x, test_y, sweep_c, kfold, is_debug):
    print(f"")
    print(f"")
    print(f"<------------ Grid Search & Test for Kernel: Linear------------>")
    svm_type_def = '-s 0 -t 0'
    param        = svm_parameter(svm_type_def)
    print(f"svm_type = {param.svm_type}, kernel_type = {param.kernel_type}, kfold = {kfold}")

    #grid search
    best_acc = -1
    best_c = -1

    for c_value in sweep_c:
        param = svm_parameter(svm_type_def+' -c '+str(c_value)+' -v '+str(kfold)+' -q')
        prob  = svm_problem(train_y, train_x)

        if(is_debug):
            print(f"param.svm_type = {param.svm_type}, param.kernel_type = {param.kernel_type}, param.C = {param.C}, kfold = {kfold}")

        accuracy = svm_train(prob, param)

        if(accuracy > best_acc):
            best_acc = accuracy
            best_c   = c_value
            print(f"Currently best C = {best_c}, accuracy = {best_acc}%")

    print(f"> --------------------------------------------------------------")
    print(f"> Done for grid search: ")
    print(f"Best hyper parameter C = {best_c}, with highest accuracy = {best_acc}%")
    print(f"> --------------------------------------------------------------")

    print(f"> Testing with test datasets: ")

    #testing
    prob  = svm_problem(train_y, train_x) #retrain the model with opt hyper parameters
    param = svm_parameter(svm_type_def+' -c '+str(best_c)+' -q')
    model = svm_train(prob, param)
    svm_predict(test_y, test_x, model)

    print(f"<------------------------ END ------------------------------>")

def GridSearchKernelPolynomial(train_x, train_y, test_x, test_y, sweep_c, sweep_g, sweep_c0, sweep_d, kfold, is_debug):
    print(f"")
    print(f"")
    print(f"<------------ Grid Search & Test for Kernel: Polynomial------------>")
    svm_type_def = '-s 0 -t 1'
    param        = svm_parameter(svm_type_def)
    print(f"svm_type = {param.svm_type}, kernel_type = {param.kernel_type}, kfold = {kfold}")

    #grid search
    best_acc = -1
    best_c   = -1
    best_g   = -1
    best_c0  = -1
    best_d   = -1

    for c_value in sweep_c:
        for g_value in sweep_g:
            for c0_value in sweep_c0:
                for d_value in sweep_d:
                    param = svm_parameter(svm_type_def+' -c '+str(c_value)+' -g '+str(g_value)+' -r '+str(c0_value)+' -d '+str(d_value)+' -v '+str(kfold)+' -q')
                    prob  = svm_problem(train_y, train_x)

                    if(is_debug):
                        print(f"param.svm_type = {param.svm_type}, param.kernel_type = {param.kernel_type}, param.C = {param.C}, param.gamma = {param.gamma}, param.coef0 = {param.coef0}, param.degree = {param.degree}, kfold = {kfold}")

                    accuracy = svm_train(prob, param)

                    if(accuracy > best_acc):
                        best_acc = accuracy
                        best_c   = c_value
                        best_g   = g_value
                        best_c0  = c0_value
                        best_d   = d_value
                        print(f"Currently best C = {best_c}, best gamma = {best_g}, best coef0 = {best_c0}, best degree = {best_d}, accuracy = {best_acc}%")

    print(f"> --------------------------------------------------------------")
    print(f"> Done for grid search: ")
    print(f"Best hyper parameter C = {best_c}, gamma = {best_g}, coef0 = {best_c0}, degree = {best_d}, with highest accuracy = {best_acc}%")
    print(f"> --------------------------------------------------------------")

    print(f"> Testing with test datasets: ")

    #testing
    prob  = svm_problem(train_y, train_x) #retrain the model with opt hyper parameters
    param = svm_parameter(svm_type_def+' -c '+str(best_c)+' -g '+str(best_g)+' -r '+str(best_c0)+' -d '+str(best_d)+' -q')
    model = svm_train(prob, param)
    svm_predict(test_y, test_x, model)

    print(f"<------------------------ END ------------------------------>")

def GridSearchKernelRBF(train_x, train_y, test_x, test_y, sweep_c, sweep_g, kfold, is_debug):
    print(f"")
    print(f"")
    print(f"<------------ Grid Search & Test for Kernel: RBF------------>")
    svm_type_def = '-s 0 -t 2'
    param        = svm_parameter(svm_type_def)
    print(f"svm_type = {param.svm_type}, kernel_type = {param.kernel_type}, kfold = {kfold}")

    #grid search
    best_acc = -1
    best_c   = -1
    best_g   = -1

    for c_value in sweep_c:
        for g_value in sweep_g:
            param = svm_parameter(svm_type_def+' -c '+str(c_value)+' -g '+str(g_value)+' -v '+str(kfold)+' -q')
            prob  = svm_problem(train_y, train_x)

            if(is_debug):
                print(f"param.svm_type = {param.svm_type}, param.kernel_type = {param.kernel_type}, param.C = {param.C}, param.gamma = {param.gamma}, kfold = {kfold}")

            accuracy = svm_train(prob, param)

            if(accuracy > best_acc):
                best_acc = accuracy
                best_c   = c_value
                best_g   = g_value
                print(f"Currently best C = {best_c}, best gamma = {best_g}, accuracy = {best_acc}%")

    print(f"> --------------------------------------------------------------")
    print(f"> Done for grid search: ")
    print(f"Best hyper parameter C = {best_c}, gamma = {best_g}, with highest accuracy = {best_acc}%")
    print(f"> --------------------------------------------------------------")

    print(f"> Testing with test datasets: ")

    #testing
    prob  = svm_problem(train_y, train_x) #retrain the model with opt hyper parameters
    param = svm_parameter(svm_type_def+' -c '+str(best_c)+' -g '+str(best_g)+' -q')
    model = svm_train(prob, param)
    svm_predict(test_y, test_x, model)

    print(f"<------------------------ END ------------------------------>")

def GridSearchOpt(train_x, train_y, test_x, test_y, kfold, search_lin, search_pol, search_rbf, is_debug):
    train_y = train_y.ravel()
    test_y  = test_y.ravel()

    if(search_lin):
        sweep_c = []
        for i in range(-13, 14):
            sweep_c.append(np.power(2.0, i))
        GridSearchKernelLinear(train_x, train_y, test_x, test_y, sweep_c, kfold, is_debug)

    if(search_pol):
        sweep_c  = []
        sweep_g  = [float(1/784)]
        sweep_c0 = [0]
        sweep_d  = [0]
        for i in range(-13, 14):
            sweep_c.append(np.power(2.0, i))
        for i in range(-13, 14):
            sweep_g.append(np.power(2.0, i))
        for i in range(-13, 14):
            sweep_c0.append(np.power(2.0, i))
        for i in range(1, 11):
            sweep_d.append(i)
        GridSearchKernelPolynomial(train_x, train_y, test_x, test_y, sweep_c, sweep_g, sweep_c0, sweep_d, kfold, is_debug)

    if(search_rbf):
        sweep_c  = []
        sweep_g  = [float(1/784)]
        for i in range(-13, 14):
            sweep_c.append(np.power(2.0, i))
        for i in range(-13, 14):
            sweep_g.append(np.power(2.0, i))
        GridSearchKernelRBF(train_x, train_y, test_x, test_y, sweep_c, sweep_g, kfold, is_debug)

def SVMModelComparison(train_x, train_y, test_x, test_y):
    train_y = train_y.ravel()
    test_y  = test_y.ravel()

    kernel_type = {0:"Linear Kernel", 1:"Polynomial Kernel", 2:"RBF Kernel"}
    kernel_parameter = {0:"None", 1:["gamma", "coef0", "degree"], 2:["gamma"]}

    prob = svm_problem(train_y, train_x)
    for kernel_index in range(3):
        if(kernel_index==0):
            print(f"------------")

        #setting the input and the parameters
        param = svm_parameter('-t '+str(kernel_index)+' -q')

        ## training  the model
        model = svm_train(prob, param)

        #testing the model
        print(f"{kernel_type[kernel_index]}: ")
        if(kernel_parameter[kernel_index] != 'None'):
            for parameter_name in kernel_parameter[kernel_index]:
                if(parameter_name == "gamma"):
                    print(f"parameter gamma = {param.gamma}")
                elif(parameter_name == "coef0"):
                    print(f"parameter coef0 = {param.coef0}")
                elif(parameter_name == "degree"):
                    print(f"parameter degree = {param.degree}")

        print(f"")
        svm_predict(test_y, test_x, model)
        print(f"------------")

def PrintMatrix(input_matrix, matrix_name):
    print(f'{len(input_matrix)}x{len(input_matrix[0])}, {matrix_name}: ')
#    print(f'[', end = '')
    for index_i, rows in enumerate(input_matrix):
        for index_j, cols in enumerate(rows):
            if(index_i == (len(input_matrix)-1) and index_j == (len(rows)-1)):
                print(f'{input_matrix[index_i][index_j]:20.10f}') #will print the same
                #print(f'[{cols}] ') #will print the same
            elif(index_j == (len(rows)-1)):
                print(f'{input_matrix[index_i][index_j]:20.10f}') #will print the same
            else:
                if(index_j == 0 and index_i != 0):
                    print(f'{input_matrix[index_i][index_j]:20.10f}', end='') #will print the same
                else:
                    print(f'{input_matrix[index_i][index_j]:20.10f}', end='') #will print the same

#---------------Execution---------------#
if __name__ == '__main__':
    main()
