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
    (m, s, gaussian_meth, EPSILON_m, EPSILON_s, is_debug) = ArgumentParser()

    #Perform Sequential Estimator
    SequentialEstimation(EPSILON_m, EPSILON_s, m, s, gaussian_meth)

    if(is_debug):
        DrawGaussianDistribution(m, s, 1)
        DrawGaussianDistribution(m, s, 0)

#########################
#     Sub-Routine       #
#########################
def ArgumentParser():
    m                 = None
    s                 = None
    gaussian_meth     = None
    EPSILON_m         = pow(10, -8)
    EPSILON_s         = pow(10, -8)
    is_debug          = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--m", "-m", help="The mean for univariate Gaussian data generator.")
    parser.add_argument("--s", "-s", help="The variance for univariate Gaussian data generator.")
    parser.add_argument("--gaussian_meth", "-gm", help="Set '0' to use 12 unform distribution to approximate standard Gaussian. Set '1' to use Box-Muller method to generate standard Gaussian. Default is '0'.")
    parser.add_argument("--EPSILON_m", "-EPS_m", help="set the error of the mean of neighbored steps for terminating the Sequential Estimation. Default 1e^-8.")
    parser.add_argument("--EPSILON_s", "-EPS_s", help="set the error of the variance of neighbored steps for terminating the Sequential Estimation. Default 1e^-8.")
    parser.add_argument("--is_debug", "-isd", help="1 for debug mode; 0 for normal mode.")

    args = parser.parse_args()

    if(args.m):
        m = float(args.m)
    if(args.s):
        s = float(args.s)
    if(args.gaussian_meth):
        gaussian_meth = int(args.gaussian_meth)
    if args.EPSILON_m:
        EPSILON_m = float(args.EPSILON_m)
    if args.EPSILON_s:
        EPSILON_s = float(args.EPSILON_s)
    if(args.is_debug):
        is_debug = int(args.is_debug)


    if(m == None):
        print(f"Error: You should set '--m' or '-m' for the input of mean for univariate Gaussian data generator.")
        sys.exit()

    if(s == None):
        print(f"Error: You should set '--s' or '-s' for the input of variance for univariate Gaussian data generator.")
        sys.exit()

    if(gaussian_meth == None):
        gaussian_meth = 0

    if(is_debug):
        print(f"m = {m}")
        print(f"s = {s}")
        print(f"gaussian_meth = {gaussian_meth}")

    return (m, s, gaussian_meth, EPSILON_m, EPSILON_s, is_debug)

def DrawGaussianDistribution(m, s, method):
    xaxis_range = np.arange(-10000, 10000, 1)
    yaxis_range = [UnivariateGaussianRandomGenerator(m, s, method) for x in xaxis_range]
    plt.hist(yaxis_range, 100)
    plt.title(f"Gaussian distribution using method {method}")
    plt.show()

def UnivariateGaussianRandomGenerator(m, s, method=0):
    data_point = 0.0

    if(method==0):
        data_point = (sum(np.random.uniform(0, 1, 12)) - 6) * math.sqrt(s) + m
    else:
        u_dist = np.random.uniform(0, 1)
        v_dist = np.random.uniform(0, 1)
        data_point = (math.sqrt((-2)*math.log(u_dist)) * math.sin(2*math.pi*v_dist)) * math.sqrt(s) + m

    return data_point

def WelfordOnlineAlg(new_data_point, last_count, last_mean, last_M2):
    if(last_count == 0):
        count = 1
        mean  = new_data_point
        M2    = 0
        unbiased_var = 0
    else:
        count        = last_count + 1
        delta1       = new_data_point - last_mean
        mean         = last_mean + delta1/count
        delta2       = new_data_point - mean
        M2           = last_M2 + delta1*delta2
        unbiased_var = M2/(count-1)

    return (count, mean, M2, unbiased_var)

def SequentialEstimation(EPSILON_m, EPSILON_s, m, s, gaussian_meth):
    last_count = 0
    last_mean  = 0
    last_M2    = 0
    print(f"Data point source function: N({m}, {s})")
    print(f"")
    while(True):
        new_data_point                           = UnivariateGaussianRandomGenerator(m, s, gaussian_meth)
        (count, unbiased_mean, M2, unbiased_var) = WelfordOnlineAlg(new_data_point, last_count, last_mean, last_M2)
        if(last_count<2):
            last_unbiased_var = 0
        else:
            last_unbiased_var = last_M2/(last_count-1)

        print(f"Add data point: {new_data_point}")
        print(f"Mean = {unbiased_mean}\tVariance = {unbiased_var}")

        if((abs(unbiased_mean-last_mean) < EPSILON_m) and (abs(unbiased_var-last_unbiased_var) < EPSILON_s)):
            break

        last_count = count
        last_mean  = unbiased_mean
        last_M2    = M2


#---------------Execution---------------#
if __name__ == '__main__':
    main()
