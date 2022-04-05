#! /usr/bin/env python3
'''
    Author      : BCC
    Date        : 2022/03/31
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
    (b, n, w, a, gaussian_meth, is_debug) = ArgumentParser()

    #Generate a random data point from Polynomial Basis Linear Model Data Generator
    poly_point = PolynomialBasisRandomGenerator(n, w, a, gaussian_meth)

    if(is_debug):
        print(f'b = {b}')
        print(f'n = {n}')
        print(f'a = {a}')
        print(f'w = {w}')
        print(f'gaussian_meth = {gaussian_meth}')
        print(f"poly_point : ({poly_point[0]}, {poly_point[1]})")

#########################
#     Sub-Routine       #
#########################
def ArgumentParser():
    b                 = None
    n                 = None
    w                 = None
    a                 = None
    gaussian_meth     = None
    is_debug          = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--b", "-b", help="The precision for initital prior p(w) ~ N(0, b^-1*I).")
    parser.add_argument("--n", "-n", help="The basis number of Polynomial basis linear model data generator.")
    parser.add_argument("--w", "-w", help="The n coefficient of Polynomial basis linear model data generator.")
    parser.add_argument("--a", "-a", help="The variance of noise of Polynomial basis linear model data generator.")
    parser.add_argument("--gaussian_meth", "-gm", help="Set '0' to use 12 unform distribution to approximate standard Gaussian. Set '1' to use Box-Muller method to generate standard Gaussian. Default is '0'.")
    parser.add_argument("--is_debug", "-isd", help="1 for debug mode; 0 for normal mode.")

    args = parser.parse_args()

    if(args.b):
        b = float(args.b)
    if(args.n):
        n = int(args.n)
    if(args.w):
        w = ConvertToList(args.w)
    if(args.a):
        a = float(args.a)
    if(args.gaussian_meth):
        gaussian_meth = int(args.gaussian_meth)
    if(args.is_debug):
        is_debug = int(args.is_debug)

    if(b == None):
        print(f"Error: You should set '--b' or '-b' for the input of variance for univariate Gaussian data generator.")
        sys.exit()

    if(n == None):
        print(f"Error: You should set '--n' or '-n' for the input of the number of basis for Polynomial basis linear model data generator.")
        sys.exit()

    if(w == None):
        print(f"Error: You should set '--w' or '-w' for the input of the vector W for Polynomial basis linear model data generator.")
        sys.exit()

    if(a == None):
        print(f"Error: You should set '--a' or '-a' for the input of the variance of noise e for Polynomial basis linear model data generator.")
        sys.exit()

    if(gaussian_meth == None):
        gaussian_meth = 0

    return (b, n, w, a, gaussian_meth, is_debug)

    '''
def DrawGaussianDistribution(m, s, method):
    xaxis_range = np.arange(-10000, 10000, 1)
    yaxis_range = [UnivariateGaussianRandomGenerator(m, s, method) for x in xaxis_range]
    plt.hist(yaxis_range, 100)
    plt.title(f"Gaussian distribution using method {method}")
    plt.show()
    '''

def UnivariateGaussianRandomGenerator(m, s, method=0):
    data_point = 0.0

    if(method==0):
        data_point = (sum(np.random.uniform(0, 1, 12)) - 6) * math.sqrt(s) + m
    else:
        u_dist = np.random.uniform(0, 1)
        v_dist = np.random.uniform(0, 1)
        data_point = (math.sqrt((-2)*math.log(u_dist)) * math.sin(2*math.pi*v_dist)) * math.sqrt(s) + m

    return data_point

def PolynomialBasisRandomGenerator(n, w, a, gaussian_meth):
    x = np.random.uniform(-1, 1)
    y = 0
    tmp_x = 1
    for i in range(n):
        if(i == 0):
            y += w[i]
        else:
            tmp_x *= x
            y += w[i]*tmp_x

    y += UnivariateGaussianRandomGenerator(0, a, gaussian_meth)
    return (x, y)

def ConvertToList(w_string):
    return (list(map(float, w_string.strip('[]').split(','))))

#---------------Execution---------------#
if __name__ == '__main__':
    main()
