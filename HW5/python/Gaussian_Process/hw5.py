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
import gzip
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.optimize import minimize
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
    print(f"> ArgumentParser...")
    (input_file, alpha, length_scale, sigma, use_sigma, beta, test_num, opt_hyper, is_debug) = ArgumentParser()

    #Get the input data point
    print(f"> ReadInputfile...")
    input_data = ReadInputFile(input_file)

    #Perform Gaussian Process
    print(f"> GaussianProcess...")
    GaussianProcess(input_data, alpha, length_scale, sigma, use_sigma, beta, test_num, opt_hyper)

    if(is_debug):
        for index, w in enumerate(input_data):
            print(f'{w[0]:25.18f}\t{w[1]:25.18f}')

#########################
#     Sub-Routine       #
#########################
def ArgumentParser():
    input_file          = None
    alpha               = 1
    length_scale        = 1
    beta                = 5
    test_num            = 500
    opt_hyper           = 0
    sigma               = 1
    use_sigma           = 0
    is_debug            = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",   "-inf",     help="The input file name of the input data points.")
    parser.add_argument("--alpha",        "-alpha",   help="The input aplha, mixture parameter for rational quadratic kernel.")
    parser.add_argument("--length_scale", "-ls",      help="The input length scale of rational quadratic kernel.")
    parser.add_argument("--beta",        "-beta",     help="The input beta, the variance of the noise Gaussian model.")
    parser.add_argument("--test_num",    "-test_num", help="The number of test data point in the range of [-60, 60]. Default 500.")
    parser.add_argument("--opt_hyper",   "-opt_hyper",help="Whether to optimize the hyper parameter, alpha and length_scale, for rational quadratic kernel. Default 0.")
    parser.add_argument("--sigma",       "-sigma",    help="The sigma value of rational quadratic kernel. Default 1.")
    parser.add_argument("--use_sigma",   "-use_sigma",help="1 for using sigam in calculating rational quadratic kernel. 0 for not using sigma incalculating rational quadratic kernel. Default 0.")
    parser.add_argument("--is_debug",    "-isd",      help="1 for debug mode; 0 for normal mode.")

    args = parser.parse_args()

    if(args.input_file):
        input_file   = args.input_file
    if(args.alpha):
        alpha        = float(args.alpha)
    if(args.length_scale):
        length_scale = float(args.length_scale)
    if(args.beta):
        beta         = float(args.beta)
    if(args.test_num):
        test_num     = int(args.test_num)
    if(args.opt_hyper):
        opt_hyper    = int(args.opt_hyper)
    if(args.sigma):
        sigma        = float(args.sigma)
    if(args.use_sigma):
        use_sigma    = int(args.use_sigma)
    if(args.is_debug):
        is_debug     = int(args.is_debug)

    if(input_file == None):
        print(f"Error: You should set '--iput_file' or '-inf' for the input file name of trainin data.")
        sys.exit()

    if(is_debug):
        print(f"input_file   = {input_file}")
        print(f"alpha        = {alpha}")
        print(f"length_scale = {length_scale}")
        print(f"beta         = {beta}")
        print(f"test_num     = {test_num}")
        print(f"opt_hyper    = {opt_hyper}")
        print(f"sigma        = {sigma}")
        print(f"use_sigma    = {use_sigma}")
        print(f"is_debug     = {is_debug}")

    return (input_file, alpha, length_scale, sigma, use_sigma, beta, test_num, opt_hyper, is_debug)

def ReadInputFile(input_file):
    file_data  = open(input_file, 'r')
    file_lines = file_data.readlines()
    input_data = []

    for line in file_lines:
        m = re.match(r"\s*(\S+) (\S+)\s*", line)
        if(m is not None):
            data_pairs = []
            data_pairs.append(float(m.group(1)))
            data_pairs.append(float(m.group(2)))
            input_data.append(data_pairs)

    return input_data

def RationalQuadraticKernelMatrix(xi_m, xj_m, alpha, length_scale, use_sigma, beta, sigma=1, add_beta=0): #xi_m: nx1 matrix, xj_m: nx1 matrix
    row = xi_m.shape[0]
    col = xj_m.shape[0]
    kernel_mat = []

    dist_matrix = distance.cdist(xi_m, xj_m, 'euclidean')

    for i in range(row):
        row_arr = []
        for j in range(col):
            if(use_sigma):
                kernel_val = math.pow(sigma, 2)*math.pow(1 + (math.pow(dist_matrix[i][j], 2)/(2*alpha*math.pow(length_scale, 2))), -1*alpha)
            else:
                kernel_val = math.pow(1 + (math.pow(dist_matrix[i][j], 2)/(2*alpha*math.pow(length_scale, 2))), -1*alpha)
            if((i==j) and (add_beta==1)):
                kernel_val += 1/beta

            row_arr.append(kernel_val)
        kernel_mat.append(row_arr)

    return np.array(kernel_mat)

def NegMarginalLogLikeliHoodUseSigma(hyper_param, x_data, y_data, use_sigma, beta):
    N = x_data.shape[0]
    c_train_mat = RationalQuadraticKernelMatrix(x_data, x_data, hyper_param[0], hyper_param[1], use_sigma, beta, hyper_param[2], 1)
    result = 0.5*np.log(np.linalg.det(c_train_mat)) + 0.5*((y_data.T@np.linalg.inv(c_train_mat))@y_data) + 0.5*N*np.log(2*np.pi)

    return result

def NegMarginalLogLikeliHood(hyper_param, x_data, y_data, beta):
    N = x_data.shape[0]
    c_train_mat = RationalQuadraticKernelMatrix(x_data, x_data, hyper_param[0], hyper_param[1], 0, beta, 1, 1)
    result = 0.5*np.log(np.linalg.det(c_train_mat)) + 0.5*((y_data.T@np.linalg.inv(c_train_mat))@y_data) + 0.5*N*np.log(2*np.pi)

    return result

def GaussianProcess(input_data, alpha, length_scale, sigma, use_sigma, beta, test_num, opt_hyper):
    x_data = np.array([[x[0] for x in input_data]]).T
    y_data = np.array([[x[1] for x in input_data]]).T

    if(opt_hyper == 1):
        if(use_sigma):
            initial_hyper_param = np.array([alpha, length_scale, sigma])
            hyper_param_opt = minimize(NegMarginalLogLikeliHoodUseSigma, initial_hyper_param, args = (x_data, y_data, use_sigma, beta))
            alpha           = hyper_param_opt.x[0]
            length_scale    = hyper_param_opt.x[1]
            sigma           = hyper_param_opt.x[2]
        else:
            initial_hyper_param = np.array([alpha, length_scale])
            hyper_param_opt = minimize(NegMarginalLogLikeliHood, initial_hyper_param, args = (x_data, y_data, beta))
            alpha           = hyper_param_opt.x[0]
            length_scale    = hyper_param_opt.x[1]

    #Calculate the C(nxn) = k(x, x) + 1/beta(i==j)
    c_train_mat = RationalQuadraticKernelMatrix(x_data, x_data, alpha, length_scale, use_sigma, beta, sigma, 1)
    c_train_inv_mat = np.linalg.inv(c_train_mat)

    #Calculate k(x, x*), nxm
    x_test = np.array([np.linspace(-60, 60, test_num)]).T #mx1
    k_train_test = RationalQuadraticKernelMatrix(x_data, x_test, alpha, length_scale, use_sigma, beta, sigma, 0)

    #Calculate k_star = k* = k(x*, x*) + 1/beta(i==j), mxm
    k_star = RationalQuadraticKernelMatrix(x_test, x_test, alpha, length_scale, use_sigma, beta, sigma, 1)

    #Calculate mean and variance
    mean = (k_train_test.T)@(c_train_inv_mat@y_data) #mx1
    var  = k_star - (k_train_test.T)@(c_train_inv_mat@k_train_test) #mxm

    #Visualization
    Visualization(x_data, y_data, x_test, mean, var, alpha, length_scale, beta, sigma, use_sigma, test_num)

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

def Visualization(x_data, y_data, x_test, mean, var, alpha, length_scale, beta, sigma, use_sigma, test_num):
    #creat the subplot object
    fig=plt.subplots(1,1, figsize=(15, 8))

    #======================Ground truth========================#
    plt.subplot(1, 1, 1)
    plt.xlim(-60, 60)

    #plot the original data point
    if(use_sigma):
        plt.title(f"Gaussian Process with sigma = {sigma}, alpha = {alpha}, length_scale = {length_scale}, beta = {beta}, test_num = {test_num}")
    else:
        plt.title(f"Gaussian Process with alpha = {alpha}, length_scale = {length_scale}, beta = {beta}, test_num = {test_num}")
    plt.scatter(x_data, y_data, c='m')

    #plot the predicton result
    plt.plot(x_test.T[0], mean.T[0], c='blue')

    #95% confidence interval
    var_x = np.diag(var)
    high = np.array([x + 1.96*math.sqrt(var_x[index]) for index, x in enumerate(mean.T[0])])
    low  = np.array([x - 1.96*math.sqrt(var_x[index]) for index, x in enumerate(mean.T[0])])
    plt.fill_between(x_test.T[0], high, low, color='red', alpha=0.2)

    #show the plot
    plt.tight_layout()
    plt.show()

def ConvertToList(w_string):
    return (list(map(float, w_string.strip('[]').split(','))))

#---------------Execution---------------#
if __name__ == '__main__':
    main()
