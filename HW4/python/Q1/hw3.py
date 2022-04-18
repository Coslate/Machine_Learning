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
    (N, mx1, vx1, my1, vy1, mx2, vx2, my2, vy2, gaussian_meth, is_debug) = ArgumentParser()

    #Generate a random data point from Univariate Gaussian distribution
    uni_gauss_point_x1 = UnivariateGaussianRandomGenerator(mx1, vx1, gaussian_meth)
    uni_gauss_point_y1 = UnivariateGaussianRandomGenerator(my1, vy1, gaussian_meth)
    uni_gauss_point_x2 = UnivariateGaussianRandomGenerator(mx2, vx2, gaussian_meth)
    uni_gauss_point_y2 = UnivariateGaussianRandomGenerator(my2, vy2, gaussian_meth)

    print(f"uni_gauss_point_x1 = {uni_gauss_point_x1}")
    print(f"uni_gauss_point_y1 = {uni_gauss_point_y1}")
    print(f"uni_gauss_point_x2 = {uni_gauss_point_x2}")
    print(f"uni_gauss_point_y2 = {uni_gauss_point_y2}")

    if(is_debug):
        '''
        DrawGaussianDistribution(mx1, vx1, 1)
        DrawGaussianDistribution(my1, vy1, 1)
        DrawGaussianDistribution(mx2, vx2, 1)
        DrawGaussianDistribution(my2, vy2, 1)
        '''

#########################
#     Sub-Routine       #
#########################
def ArgumentParser():
    N                   = None
    mx1                 = None
    vx1                 = None
    my1                 = None
    vy1                 = None
    mx2                 = None
    vx2                 = None
    my2                 = None
    vy2                 = None
    gaussian_meth       = None
    is_debug            = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--N", "-N", help="The number of data points.")
    parser.add_argument("--mx1", "-mx1", help="The mean for univariate Gaussian data generator for point x1.")
    parser.add_argument("--vx1", "-vx1", help="The variance for univariate Gaussian data generator for point x1.")
    parser.add_argument("--my1", "-my1", help="The mean for univariate Gaussian data generator for point y1.")
    parser.add_argument("--vy1", "-vy1", help="The variance for univariate Gaussian data generator for point y1.")
    parser.add_argument("--mx2", "-mx2", help="The mean for univariate Gaussian data generator for point x2.")
    parser.add_argument("--vx2", "-vx2", help="The variance for univariate Gaussian data generator for point x2.")
    parser.add_argument("--my2", "-my2", help="The mean for univariate Gaussian data generator for point y2.")
    parser.add_argument("--vy2", "-vy2", help="The variance for univariate Gaussian data generator for point y2.")
    parser.add_argument("--gaussian_meth", "-gm", help="Set '0' to use 12 unform distribution to approximate standard Gaussian. Set '1' to use Box-Muller method to generate standard Gaussian. Default is '0'.")
    parser.add_argument("--is_debug", "-isd", help="1 for debug mode; 0 for normal mode.")

    args = parser.parse_args()

    if(args.N):
        N   = int(args.N)
    if(args.mx1):
        mx1 = float(args.mx1)
    if(args.vx1):
        vx1 = float(args.vx1)
    if(args.my1):
        my1 = float(args.my1)
    if(args.vy1):
        vy1 = float(args.vy1)
    if(args.mx2):
        mx2 = float(args.mx2)
    if(args.vx2):
        vx2 = float(args.vx2)
    if(args.my2):
        my2 = float(args.my2)
    if(args.vy2):
        vy2 = float(args.vy2)
    if(args.gaussian_meth):
        gaussian_meth = int(args.gaussian_meth)
    if(args.is_debug):
        is_debug = int(args.is_debug)

    if(N == None):
        print(f"Error: You should set '--N' or '-N' for the number of data points.")
        sys.exit()

    if(mx1 == None):
        print(f"Error: You should set '--mx1' or '-mx1' for the input of mean for univariate Gaussian data generator for point x1.")
        sys.exit()

    if(vx1 == None):
        print(f"Error: You should set '--vx1' or '-vx1' for the input of variance for univariate Gaussian data generator for point x1.")
        sys.exit()

    if(my1 == None):
        print(f"Error: You should set '--my1' or '-my1' for the input of mean for univariate Gaussian data generator for point y1.")
        sys.exit()

    if(vy1 == None):
        print(f"Error: You should set '--vy1' or '-vy1' for the input of variance for univariate Gaussian data generator for point x1.")
        sys.exit()

    if(mx2 == None):
        print(f"Error: You should set '--mx2' or '-mx2' for the input of mean for univariate Gaussian data generator for point x2.")
        sys.exit()

    if(vx2 == None):
        print(f"Error: You should set '--vx2' or '-vx2' for the input of variance for univariate Gaussian data generator for point x2.")
        sys.exit()

    if(my2 == None):
        print(f"Error: You should set '--my2' or '-my2' for the input of mean for univariate Gaussian data generator for point y2.")
        sys.exit()

    if(vy2 == None):
        print(f"Error: You should set '--vy2' or '-vy2' for the input of variance for univariate Gaussian data generator for point x2.")
        sys.exit()

    if(gaussian_meth == None):
        gaussian_meth = 0

    if(is_debug):
        print(f"N   = {N}")
        print(f"mx1 = {mx1}")
        print(f"vx1 = {vx1}")
        print(f"my1 = {my1}")
        print(f"vy1 = {vy1}")
        print(f"mx2 = {mx2}")
        print(f"vx2 = {vx2}")
        print(f"my2 = {my2}")
        print(f"vy2 = {vy2}")
        print(f"gaussian_meth = {gaussian_meth}")

    return (N, mx1, vx1, my1, vy1, mx2, vx2, my2, vy2, gaussian_meth, is_debug)

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

#---------------Execution---------------#
if __name__ == '__main__':
    main()
