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
    (b, n, w, a, gaussian_meth, EPSILON, is_debug) = ArgumentParser()

    PerformBayesianLinearRegression(n, w, a, b, gaussian_meth, EPSILON)

    if(is_debug):
        print(f'b = {b}')
        print(f'n = {n}')
        print(f'a = {a}')
        print(f'w = {w}')
        print(f'gaussian_meth = {gaussian_meth}')

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
    EPSILON           = pow(10, -8)

    parser = argparse.ArgumentParser()
    parser.add_argument("--b", "-b", help="The precision for initital prior p(w) ~ N(0, b^-1*I).")
    parser.add_argument("--n", "-n", help="The basis number of Polynomial basis linear model data generator.")
    parser.add_argument("--w", "-w", help="The n coefficient of Polynomial basis linear model data generator.")
    parser.add_argument("--a", "-a", help="The variance of noise of Polynomial basis linear model data generator.")
    parser.add_argument("--gaussian_meth", "-gm", help="Set '0' to use 12 unform distribution to approximate standard Gaussian. Set '1' to use Box-Muller method to generate standard Gaussian. Default is '0'.")
    parser.add_argument("--epsilon", "-eps", help="set the error precision of the neighbored steps for terminating the Bayesian Linear Regression. Default 1e^-8.")
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
    if args.epsilon:
        EPSILON = float(args.epsilon)
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

    return (b, n, w, a, gaussian_meth, EPSILON, is_debug)

    '''
def DrawGaussianDistribution(m, s, method):
    xaxis_range = np.arange(-10000, 10000, 1)
    yaxis_range = [UnivariateGaussianRandomGenerator(m, s, method) for x in xaxis_range]
    plt.hist(yaxis_range, 100)
    plt.title(f"Gaussian distribution using method {method}")
    plt.show()
    '''

def PerformBayesianLinearRegression(n, w, a, b, gaussian_meth, EPSILON):
    all_data_points = []
    count           = 0
    prior_m         = MatrixTranspose([[0 for x in range(n)]]) #0 vector, nx1
    prior_S         = MatrixIdentityGen(n, b) #bI, nxn

    while(True):
        #Generate a random data point from Polynomial Basis Linear Model Data Generator
        poly_point = PolynomialBasisRandomGenerator(n, w, a, gaussian_meth)
        all_data_points.append(poly_point)

        #Form the design matrix & Y
        (design_matrix, Y) = FormMatrix([poly_point], n)

        #Calculate the Posterior
        posterior_S = MatrixAdd(MatrixMulScalar(MatrixMul(MatrixTranspose(design_matrix), design_matrix), a), prior_S) # a*(design_matrix^T*design_matrix) + S
        posterior_m = MatrixMul(MatrixInverse(posterior_S), MatrixAdd(MatrixMulScalar(MatrixMul(MatrixTranspose(design_matrix), Y), a), MatrixMul(prior_S, prior_m))) # posterior_S^-1 * (a*(design_matrix^T*Y) + S*m)
        posterior_V = MatrixInverse(posterior_S)

        #Calculate error: ||posterior_m-prior_m||^2
        error_try   = ErrorVectorCalculation(MatrixSub(posterior_m, prior_m))

        print(f"Add data point ({poly_point[0]}, {poly_point[1]}) : ")
        print(f"")
        PrintMatrix(posterior_m, "Posterior mean")
        print(f"")
        PrintMatrix(posterior_V, "Posterior variance")
        print(f"")

        predict_m = MatrixMul(design_matrix, posterior_m) #design_matrix*posterior_m
        predict_V = (1/a) + MatrixMul(MatrixMul(design_matrix, posterior_V), MatrixTranspose(design_matrix))[0][0] #1/a + design_matrix*posterior_V*design_matrix^T
        print(f"Predictive distribution ~ N({predict_m[0][0]}, {predict_V})")
        print(f"")

        if(error_try < EPSILON):
            Visualization(lse_parameter_x, newton_parameter_x, input_data, error_val_lse, error_val_newton, is_debug)
            break

        #Prior Update
        prior_m = copy.deepcopy(posterior_m)
        prior_S = copy.deepcopy(posterior_S)

def Visualization(a, w, all_data_points, predict_m, predict_V, is_debug):
    #creat the subplot object
    fig=plt.subplots(1,2)

    #======================Ground truth========================#
    plt.subplot(2, 1, 1)
    plt.xlim(-2, 2)
    plt.ylim(-25, 25)

    #setting the scale for separating x, y
    x_major_locator=MultipleLocator(1)
    y_major_locator=MultipleLocator(10)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)

    #plot the input_data points in red
    x = [rows[0] for rows in input_data]
    y = [rows[1] for rows in input_data]
    plt.title("Ground truth")
    plt.scatter(x, y, c='blue', s=15, zorder=3)

    #plot the fitting line
    fitted = np.poly1d(w.reverse())
    xaxis_range = np.arange(-2, 2, 0.01)
    yaxis_range = fitted(xaxis_range)
    plt.plot(xaxis_range, yaxis_range, color='black')
    if(is_debug):
        print(f"LSE, xaxis_range = {xaxis_range}")
        print(f"LSE, yaxis_range = {yaxis_range}")
        print(f"================================")

    #======================Predict result========================#
    '''
    plt.subplot(2, 1, 2)
    plt.xlim(-6, 6)
    plt.ylim(-20, 110)

    #setting the scale for separating x, y
    x_major_locator=MultipleLocator(2)
    y_major_locator=MultipleLocator(20)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)

    #plot the input_data points in red
    x = [rows[0] for rows in input_data]
    y = [rows[1] for rows in input_data]
    plt.title("Newton Method")
    plt.scatter(x, y, c='red', s=15, zorder=3)

    #plot the fitting line
    fitted = np.poly1d([rows[0] for rows in newton_parameter_x])
    xaxis_range = np.arange(-7, 7, 0.01)
    yaxis_range = fitted(xaxis_range)
    plt.plot(xaxis_range, yaxis_range)
    if(is_debug):
        print(f"Newton, xaxis_range = {xaxis_range}")
        print(f"Newton, yaxis_range = {yaxis_range}")
        print(f"================================")

    #show the plot
    plt.show()
    '''


def ErrorVectorCalculation(vector_x):
    return (math.sqrt(MatrixMul(MatrixTranspose(vector_x), vector_x)[0][0])) # sqrt(||x||^2)

def MatrixInverse(matrix_a):
#    matrix_c = np.linalg.inv(np.array(matrix_a))
#    matrix_c.tolist()
#    x = matrix_c

    row_a = len(matrix_a)
    col_a = len(matrix_a[0])
    y     = []
    x     = [] #The inverse matrix

    if(row_a != col_a):
        print(f"Error: Input matrix: {matrix_a} is not a square matrix, and cannot be found an inverse through LU Decomposition.")
        sys.exit()
    if(DeterminationCal(matrix_a, row_a, col_a) == 0):
        print(f"Error: The det of matrix: {matrix_a} is 0, so it is not invertible.")
        sys.exit()


    #Initialization of y and x
    for i in range(row_a):
        row_y = []
        row_x = []
        for j in range(col_a):
            row_y.append(0)
            row_x.append(0)

        y.append(row_y)
        x.append(row_x)


    #Perform the LU Decomposition
    (matrix_a_l, matrix_a_u) = LUDecomposition(matrix_a)

    #Perform the inverse - first step  : find the y of Ly = [e1 e2 e3], where e1, e2, e3 are columns of I
    for i in range(row_a):
        for j in range(col_a):
            if(i==0):
                if(j==0):
                    y[i][j] = 1
                else:
                    y[i][j] = 0
            else:
                if(i==j):
                    tmp_y = 0
                    for k in range(i):
                        tmp_y += matrix_a_l[i][k]*y[k][j]
                    y[i][j] = 1 - tmp_y
                else:
                    tmp_y = 0
                    for k in range(i):
                        tmp_y += matrix_a_l[i][k]*y[k][j]
                    y[i][j] = -tmp_y

    #Perform the inverse - second step : find the x of Ux = [y1 y2 y3], where y1, y2, y3 are columns of y
    for i in range((row_a-1), -1, -1):
        for j in range(col_a):
            if(i == (row_a-1)):
                x[i][j] = y[i][j]/matrix_a_u[i][i]
            else:
                tmp_x = 0
                for k in range((row_a-1), i, -1):
                    tmp_x += x[k][j]*matrix_a_u[i][k]
                x[i][j] = (y[i][j]-tmp_x)/matrix_a_u[i][i]

    return x

def PrintMatrix(input_matrix, matrix_name):
    print(f'{matrix_name}: ')
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

def MatrixMulScalar(matrix_a, scalar=1):
    row_a = len(matrix_a)
    col_a = len(matrix_a[0])
    matrix_c = []

    #Initialization matrix_c
    for i in range(row_a):
        row = []
        for j in range(col_a):
            row.append(0)

        matrix_c.append(row)

    #Multiply every element with the scalar
    for i in range(row_a):
        for j in range(col_a):
            matrix_c[i][j] = scalar*matrix_a[i][j]

    return matrix_c

def MatrixIdentityGen(dimension, lamb=1):#lamb will multiply with 1 on the diagonal elements
    matrix_c = []

    for i in range(dimension):
        row = []
        for j in range(dimension):
            if(i==j):
                row.append(1*lamb)
            else:
                row.append(0)

        matrix_c.append(row)

    return matrix_c

def MatrixTranspose(matrix_a):
    row_a = len(matrix_a)
    col_a = len(matrix_a[0])
    matrix_c = []

    #Initialization matrix_c
    for i in range(col_a):
        row = []
        for j in range(row_a):
            row.append(0)

        matrix_c.append(row)

    for i in range(row_a):
        for j in range(col_a):
            matrix_c[j][i] = matrix_a[i][j]

    return matrix_c

def FormMatrix(input_data, poly_num):
    design_matrix = []
    b             = []

    for data_pairs in input_data:
        row_val = []
        for index in range(poly_num):
            row_val.append(pow(data_pairs[0], index))

        design_matrix.append(row_val)
        b.append([data_pairs[1]])

    return(design_matrix, b)

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

def MatrixSub(matrix_a, matrix_b):
    row_a = len(matrix_a)
    col_a = len(matrix_a[0])
    row_b = len(matrix_b)
    col_b = len(matrix_b[0])
    matrix_c = []

    if((row_a != row_b) or (col_a != col_b)):
        print(f"Error: Dimensions of matrix_a and matrix_b are different, and cannot be added together.")
        sys.exit()

    #Initialization matrix_c
    for i in range(row_a):
        row = []
        for j in range(col_a):
            row.append(0)

        matrix_c.append(row)

    #calculatations in matrix_c
    for i in range(row_a):
        for j in range(col_a):
            matrix_c[i][j] = matrix_a[i][j]-matrix_b[i][j]

    return matrix_c

def MatrixAdd(matrix_a, matrix_b):
    row_a = len(matrix_a)
    col_a = len(matrix_a[0])
    row_b = len(matrix_b)
    col_b = len(matrix_b[0])
    matrix_c = []

    if((row_a != row_b) or (col_a != col_b)):
        print(f"Error: Dimensions of matrix_a and matrix_b are different, and cannot be substracted together.")
        sys.exit()

    #Initialization matrix_c
    for i in range(row_a):
        row = []
        for j in range(col_a):
            row.append(0)

        matrix_c.append(row)

    #calculatations in matrix_c
    for i in range(row_a):
        for j in range(col_a):
            matrix_c[i][j] = matrix_a[i][j]+matrix_b[i][j]

    return matrix_c


def MatrixMul(matrix_a, matrix_b):
    row_a = len(matrix_a)
    col_a = len(matrix_a[0])
    row_b = len(matrix_b)
    col_b = len(matrix_b[0])

    if(col_a != row_b):
        print(f"Error: The number of columns of matrix_a is different with the number of rows of matrix_b, and ")
        print(f"Error: they cannot be multiplied together.")
        sys.exit()

    matrix_c = []
    row_c = row_a
    col_c = col_b

    #Initialization matrix_c
    for i in range(row_c):
        row = []
        for j in range(col_c):
            row.append(0)

        matrix_c.append(row)

    #calculatations in matrix_c
    for i in range(row_c):
        for j in range(col_c):
            tmp_result = 0
            for k in range(col_a):
                tmp_result += matrix_a[i][k]*matrix_b[k][j]

            matrix_c[i][j] = tmp_result

    return matrix_c

def SwapRow(matrix_a, row_i, row_j, num_row_a, num_col_a):
    if(row_i >= num_row_a or row_j >= num_col_a):
        print(f"Error: The input row_i or row_j is larger than the dimenssion of input matrix_a.")
        sys.exit()

    tmp_row         = copy.deepcopy(matrix_a[row_i])
    matrix_a[row_i] = copy.deepcopy(matrix_a[row_j])
    matrix_a[row_j] = copy.deepcopy(tmp_row)


def DeterminationCal(matrix_a, row_a=None, col_a=None):
    if(row_a == None):
        row_a = len(matrix_a)
    if(col_a == None):
        col_a = len(matrix_a[0])

    if(row_a != col_a):
        print(f"Error: The row and col of matrix_a are not identical, thus input metrix_a does not have det().")
        sys.exit()

    matrix_a_copy            = copy.deepcopy(matrix_a)
    have_found_row_to_change = 1
    det_result               = 0
    minus_tmp                = 1 #remember for the sign change owing to row exchange.
    for i in range(row_a):
        if(matrix_a_copy[i][i] == 0):
            have_found_row_to_change = 0
            for redund in range(i+1, row_a):
                if(matrix_a_copy[redund][i] != 0):
                    #Do the row exchange, and the result*-1
                    SwapRow(matrix_a_copy, i, redund, row_a, col_a)
                    have_found_row_to_change  = 1
                    minus_tmp                *= -1
                    break
        if(have_found_row_to_change == 0):
            det_result = 0
            break

        #Perform the row manipulation
        for redund in range(i+1, row_a):
            cancel_scalar = matrix_a_copy[redund][i]/matrix_a_copy[i][i]
            for j in range(i, col_a):
                matrix_a_copy[redund][j] = matrix_a_copy[redund][j] - cancel_scalar*matrix_a_copy[i][j]

    det_result = 1
    for i in range(row_a):
        det_result *= matrix_a_copy[i][i]
    det_result *= minus_tmp

    return det_result

def LeadingPrincipalSubmatrixFormation(matrix_a, n):
    sub_matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(matrix_a[i][j])

        sub_matrix.append(row)

    return sub_matrix


def LeadingPrincipalMinorCheck(matrix_a):
    row_a = len(matrix_a)
    col_a = len(matrix_a[0])
    det_has_zero = False
    nsub         = -1
    for i in range(row_a):
        sub_matrix = LeadingPrincipalSubmatrixFormation(matrix_a, i+1)
        det_result = DeterminationCal(sub_matrix)

        if(det_result == 0):
            det_has_zero = True
            nsub         = i+1
            break

    return (det_has_zero, nsub)


def LUDecomposition(matrix_a):
    row_a = len(matrix_a)
    col_a = len(matrix_a[0])
    matrix_l = []
    matrix_u = []

    #Check all leading principal submatrix has non-zero det.
    (det_has_zero, nsub) = LeadingPrincipalMinorCheck(matrix_a)
    if(det_has_zero):
        print(f"Error: The input matrix: {matrix_a} has 0 det in the {nsub} level of its leading principal submatrix.")
        print(f"Error: Thus, this matrix has no LU Decomposition, and the inverse by LU Decomposition fails.")
        sys.exit()


    #Initialization matrix_l, metrix_u
    for i in range(row_a):
        row1 = []
        row2 = []
        for j in range(col_a):
            row1.append(0)
            row2.append(0)

        matrix_l.append(row1)
        matrix_u.append(row2)

    #Perform the Doolittle Algorithm
    for i in range(row_a):
        for j in range(col_a):
            #Upper Triangle
            if(i==0):
                matrix_u[i][j] = matrix_a[i][j]
            elif(i>j):
                matrix_u[i][j] = 0
            else:
                tmp_for_u = 0
                for k in range(i):
                    tmp_for_u += matrix_l[i][k]*matrix_u[k][j]

                matrix_u[i][j] = matrix_a[i][j] - tmp_for_u

            #Lower Triangle
            if(j==0):
                matrix_l[i][j] = matrix_a[i][j]/matrix_u[j][j]
            elif(j==i):
                matrix_l[i][j] = 1
            elif(j>i):
                matrix_l[i][j] = 0
            else:
                tmp_for_l = 0
                for k in range(j):
                    tmp_for_l += matrix_l[i][k]*matrix_u[k][j]

                matrix_l[i][j] = (matrix_a[i][j]-tmp_for_l)/matrix_u[j][j]

    return (matrix_l, matrix_u)

#---------------Execution---------------#
if __name__ == '__main__':
    main()
