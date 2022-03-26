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
    (infile, param_a, param_b, is_debug) = ArgumentParser()

    #Get the input data point
    input_data = ReadInputFile(infile)

    #Perform the online-learning: Beta-Binomial Conjugation.
    answer_all_cases = OnlineLearningBetaBinonmialConj(input_data, param_a, param_b)

    #Print out the result
    PrintResult(answer_all_cases, param_a, param_b, input_data)

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

def BinomialDistribution(p, N, m, N_min_m):
    likelihood = (math.factorial(N)/(math.factorial(m)*math.factorial(N_min_m)))*(p**m)*((1-p)**N_min_m)
    return likelihood

def OnlineLearningBetaBinonmialConj(input_data, param_a, param_b):
    answer_all_cases = []
    prior_param_a = param_a
    prior_param_b = param_b

    for x in range(len(input_data)):
        this_result   = []
        this_result.append(prior_param_a)
        this_result.append(prior_param_b)
        total_num     = len(input_data[x])
        one_occur     = input_data[x].count(1)
        zero_occur    = input_data[x].count(0)
        theta_est_mle = one_occur/total_num
        likelihood    = BinomialDistribution(theta_est_mle, total_num, one_occur, zero_occur)

        posteriori_param_a = prior_param_a+one_occur
        posteriori_param_b = prior_param_b+zero_occur
        prior_param_a = posteriori_param_a
        prior_param_b = posteriori_param_b

        this_result.append(likelihood)
        this_result.append(posteriori_param_a)
        this_result.append(posteriori_param_b)
        answer_all_cases.append(this_result)

    return answer_all_cases

def PrintResult(answer_all_cases, param_a, param_b, input_data):
    print(f"{color.YELLOW}a = {param_a}, b = {param_b}{color.END}")
    for index, this_result in enumerate(answer_all_cases):
        input_data_str = "".join(str(x) for x in input_data[index])

        print(f"{color.UNDERLINE}case {(index+1)}: {input_data_str}{color.END}")
        print(f"Likelihood : {this_result[2]}")
        print(f"Beta prior     :    a = {this_result[0]} b = {this_result[1]}")
        print(f"Beta posterior :    a = {this_result[3]} b = {this_result[4]}")





#---------------Execution---------------#
if __name__ == '__main__':
    main()
