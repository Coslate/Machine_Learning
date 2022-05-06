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
    (input_train_x, input_train_y, input_test_x, input_test_y, is_debug) = ArgumentParser()

    #Get the input training/test data points
    #print(f"> ReadInputfile...")
    input_train_x = ReadInputFile(input_train_x)
    input_train_y = ReadInputFile(input_train_y)
    input_test_x  = ReadInputFile(input_test_x)
    input_test_y  = ReadInputFile(input_test_y)

    if(is_debug):
        '''
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
    is_debug            = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_train_x", "-input_train_x",     help="The input file name of the input training data points x.")
    parser.add_argument("--input_train_y", "-input_train_y",     help="The input file name of the input training data points y.")
    parser.add_argument("--input_test_x",  "-input_test_x",      help="The input file name of the input testing data points x.")
    parser.add_argument("--input_test_y",  "-input_test_y",      help="The input file name of the input testing data points y.")
    parser.add_argument("--is_debug",      "-isd",               help="1 for debug mode; 0 for normal mode.")

    args = parser.parse_args()

    if(args.input_train_x):
        input_train_x   = args.input_train_x
    if(args.input_train_y):
        input_train_y   = args.input_train_y
    if(args.input_test_x):
        input_test_x   = args.input_test_x
    if(args.input_test_y):
        input_test_y   = args.input_test_y
    if(args.is_debug):
        is_debug     = int(args.is_debug)

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
        print(f"is_debug        = {is_debug}")

    return (input_train_x, input_train_y, input_test_x, input_test_y, is_debug)

def ReadInputFile(input_file):
    input_data = []
    with open(input_file, 'r') as read_obj:
        csv_reader = reader(read_obj)
        # Iterate over each row in the csv using reader object
        for row in csv_reader:
            row_data = [float(element) for element in row]
            input_data.append(row_data)

    return np.array(input_data)

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
