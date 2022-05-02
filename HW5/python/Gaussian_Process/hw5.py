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
    (input_file, is_debug) = ArgumentParser()

    #Get the input data point
    input_data = ReadInputFile(input_file)


    if(is_debug):
        for index, w in enumerate(input_data):
            print(f'{w[0]:25.18f}\t{w[1]:25.18f}')

#########################
#     Sub-Routine       #
#########################
def ArgumentParser():
    input_file          = None
    is_debug            = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", "-inf", help="The input file name of the input data points.")
    parser.add_argument("--is_debug", "-isd", help="1 for debug mode; 0 for normal mode.")

    args = parser.parse_args()

    if(args.input_file):
        input_file = args.input_file

    if(args.is_debug):
        is_debug   = int(args.is_debug)

    if(input_file == None):
        print(f"Error: You should set '--N' or '-N' for the number of data points.")
        sys.exit()

    if(is_debug):
        print(f"input_file   = {input_file}")
        print(f"is_debug     = {is_debug}")

    return(input_file, is_debug)

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


def DrawGaussianDistribution(m, s, method):
    xaxis_range = np.arange(-10000, 10000, 1)
    yaxis_range = [UnivariateGaussianRandomGenerator(m, s, method) for x in xaxis_range]
    plt.hist(yaxis_range, 100)
    plt.title(f"Gaussian distribution using method {method}")
    plt.show()

def Visualization(ground_truth_d1, ground_truth_d2, grad_d1, grad_d2, newt_d1, newt_d2):
    #creat the subplot object
    fig=plt.subplots(1,3)

    #======================Ground truth========================#
    plt.subplot(1, 3, 1)
    #plt.xlim(-5, 15)
    #plt.ylim(-3, 15)

    #setting the scale for separating x, y
    #x_major_locator=MultipleLocator(10)
    #y_major_locator=MultipleLocator(2)
    #ax=plt.gca()
    #ax.xaxis.set_major_locator(x_major_locator)
    #ax.yaxis.set_major_locator(y_major_locator)

    plt.title("Ground truth")
    ground_truth_d1x = [x[0] for x in ground_truth_d1]
    ground_truth_d1y = [x[1] for x in ground_truth_d1]
    ground_truth_d2x = [x[0] for x in ground_truth_d2]
    ground_truth_d2y = [x[1] for x in ground_truth_d2]
    if((len(ground_truth_d1x) !=0) and (len(ground_truth_d1y) !=0)):
        plt.scatter(ground_truth_d1x, ground_truth_d1y, c='red')
    if((len(ground_truth_d2x) !=0) and (len(ground_truth_d2y) !=0)):
        plt.scatter(ground_truth_d2x, ground_truth_d2y, c='blue')


    #======================Gradient Descent predict result========================#
    plt.subplot(1, 3, 2)
    #plt.xlim(-5, 15)
    #plt.ylim(-3, 15)

    #setting the scale for separating x, y
    #x_major_locator=MultipleLocator(10)
    #y_major_locator=MultipleLocator(2)
    #ax=plt.gca()
    #ax.xaxis.set_major_locator(x_major_locator)
    #ax.yaxis.set_major_locator(y_major_locator)

    plt.title("Gradient descent")
    grad_d1x = [x[0] for x in grad_d1]
    grad_d1y = [x[1] for x in grad_d1]
    grad_d2x = [x[0] for x in grad_d2]
    grad_d2y = [x[1] for x in grad_d2]
    if((len(grad_d1x) !=0) and (len(grad_d1y) !=0)):
        plt.scatter(grad_d1x, grad_d1y, c='red')
    if((len(grad_d2x) !=0) and (len(grad_d2y) !=0)):
        plt.scatter(grad_d2x, grad_d2y, c='blue')

    #======================Newton's Method predict result========================#
    plt.subplot(1, 3, 3)
    #plt.xlim(-5, 15)
    #plt.ylim(-3, 15)

    #setting the scale for separating x, y
    #x_major_locator=MultipleLocator(10)
    #y_major_locator=MultipleLocator(2)
    #ax=plt.gca()
    #ax.xaxis.set_major_locator(x_major_locator)
    #ax.yaxis.set_major_locator(y_major_locator)

    plt.title("Newton's method")
    newt_d1x = [x[0] for x in newt_d1]
    newt_d1y = [x[1] for x in newt_d1]
    newt_d2x = [x[0] for x in newt_d2]
    newt_d2y = [x[1] for x in newt_d2]
    if((len(newt_d1x) !=0) and (len(newt_d1y) !=0)):
        plt.scatter(newt_d1x, newt_d1y, c='red')
    if((len(newt_d2x) !=0) and (len(newt_d2y) !=0)):
        plt.scatter(newt_d2x, newt_d2y, c='blue')

    #show the plot
    plt.tight_layout()
    plt.show()

def ConvertToList(w_string):
    return (list(map(float, w_string.strip('[]').split(','))))

#---------------Execution---------------#
if __name__ == '__main__':
    main()
