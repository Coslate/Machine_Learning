#! /usr/bin/env python3
'''
    Author      : BCC
    Date        : 2022/03/10
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
#     Main-Routine      #
#########################
def main():
    #Process the argument
    (infile, param_a, param_b, is_debug) = ArgumentParser()

    #Get the input data point
    input_data = ReadInputFile(infile)

    #Print the debug messages when necessary
    if(is_debug):
        print(f"infile     = {infile}")
        print(f"param_a    = {param_a}")
        print(f"param_b    = {param_b}")
        print(f"input_data = ")
        for x in range(len(input_data)):
            print(f"x = {x}, {input_data[x]}")
        pass


#########################
#     Sub-Routine       #
#########################
def ArgumentParser():
    param_a   = None
    param_b   = None
    infile    = None
    is_debug  = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--param_a", "-pa", help="Please set the parameter a for the initial beta prior.")
    parser.add_argument("--param_b", "-pb", help="Please set the parameter b for the initial beta prior.")
    parser.add_argument("--infile", "-inf", help="Please set the input file name for binary outcomes.")
    parser.add_argument("--is_debug", "-isd", help="1 for debug mode; 0 for normal mode.")

    args = parser.parse_args()

    if args.param_a:
        param_a = int(args.param_a)
    if args.param_b:
        param_b = int(args.param_b)
    if args.infile:
        infile = args.infile
    if args.is_debug:
        is_debug = int(args.is_debug)

    if(infile ==  None):
        print(f"Error: You should set input file name for the binary outcomes.")
        sys.exit()

    if(param_a ==  None):
        print(f"Warning: You did not set the parameter a for the initial beta prior.")
        print(f"Warning: It will be set to 0.")
        param_a = 0

    if(param_b ==  None):
        print(f"Warning: You did not set the parameter b for the initial beta prior.")
        print(f"Warning: It will be set to 0.")
        param_b = 0

    return (infile, param_a, param_b, is_debug)

def ReadInputFile(file_name):
    file_data  = open(file_name, 'r')
    file_lines = file_data.readlines()
    input_data = []

    for line in file_lines:
        row = []
        m   = re.match(r"\s*(\S+)\s*", line)
        if(m is not None):
            str_in = m.group(1)
            for x in str_in:
                row.append(int(x))

            input_data.append(row)
    return input_data

#---------------Execution---------------#
if __name__ == '__main__':
    main()
