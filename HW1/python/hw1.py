#! /usr/bin/env python3
'''
    Author      : BCC
    Date        : 2022/02/24
'''

import argparse
import math
import sys
import re
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator #for setting of scale of separating along with x-axis & y-axis.

#########################
#     Main-Routine      #
#########################
def main():
    #Process the argument
    (input_file, poly_num, lamb, is_debug, EPSILON) = ArgumentParser()

    #Get the input data point
    input_data = ReadInputFile(input_file)

    #Form the design matrix & b
    (design_matrix, b) = FormMatrix(input_data, poly_num)

    #Form the identity matrix and multiply it with lamb
    identity_matrix = MatrixIdentityGen(len(design_matrix[0]), lamb)

    #Method1 - Use LU decomposition for solving LSE regularization.
    lse_parameter_x = LSEMethod(design_matrix, identity_matrix, b)

    #CalculateError for Method1
    (error_val_lse, error_result_lse) = ErrorCalculation(design_matrix, lse_parameter_x, b)

    #Method2 - Use Newton's Method for solving LSE.
    newton_parameter_x = NewtonMethod(design_matrix, b, is_debug, EPSILON)

    #CalculateError for Method2
    (error_val_newton, error_result_newton) = ErrorCalculation(design_matrix, newton_parameter_x, b)

    #Visualization
    Visualization(lse_parameter_x, newton_parameter_x, input_data, error_val_lse, error_val_newton, is_debug)

    #Print the debug messages when necessary
    if(is_debug):
        print(f"input_file = {input_file}")
        print(f"poly_num   = {poly_num}")
        print(f"lamb       = {lamb}")

        PrintMatrix(b, "b")
        PrintMatrix(design_matrix, "design_matrix")
        print(f'raw data = ')
        for index, w in enumerate(input_data):
    #        print(f'{index}, w[0] = {str(w[0])}, w[1] = {str(w[1])}')
            print(f'{str(w[0])},{str(w[1])}')

        print(f"====================")
        print(f"Mul test:")
        test_A = [[1, 2, 3],
                [4, 5, 6],
                [2, 6, 6],
                [1, 2, 1],]

        test_B = [[1, 3],
                [7, 9],
                [13, 0]]

        (matrix_c) = MatrixMul(test_A, test_B)
        PrintMatrix(test_A, 'test_A')
        PrintMatrix(test_B, 'test_B')
        PrintMatrix(matrix_c, 'matrix_c')

        print(f"====================")
        print(f"Add test:")

        test_A = [[1, 2, 3],
                [4, 5, 6],
                [2, 6, 6]]

        test_B = [[1, 3, 99],
                [7, 9, 1],
                [13, 0, 7]]

        (matrix_c) = MatrixAdd(test_A, test_B)
        PrintMatrix(test_A, 'test_A')
        PrintMatrix(test_B, 'test_B')
        PrintMatrix(matrix_c, 'matrix_c')

        print(f"====================")
        print(f"Sub test:")

        test_A = [[1, 2, 3],
                [4, 5, 6],
                [2, 6, 6]]

        test_B = [[1, 3, 99],
                [7, 9, 1],
                [4, 88, 21]]

        (matrix_c) = MatrixSub(test_A, test_B)
        PrintMatrix(test_A, 'test_A')
        PrintMatrix(test_B, 'test_B')
        PrintMatrix(matrix_c, 'matrix_c')

        print(f"====================")
        print(f"Transpose test:")

        test_B = [[1, 3, 99],
                [7, 9, 1]]

        (matrix_c) = MatrixTranspose(test_B)
        PrintMatrix(test_B, 'test_B')
        PrintMatrix(matrix_c, 'matrix_c')

        print(f"====================")
        print(f"Identity test:")
        (matrix_c) = MatrixIdentityGen(5, 20);
        PrintMatrix(matrix_c, 'matrix_c')

        print(f"====================")
        print(f"Inverse test:")

        test_A = [[1, 2, 3],
                [4, 5, 6],
                [2, 4, 100]]

        test_B = [[1, 3, 99],
                [7, 9, 1],
                [4, 88, 21]]

        (matrix_a_inv) = MatrixInverse(test_A)
        (matrix_b_inv) = MatrixInverse(test_B)
        PrintMatrix(test_A, 'test_A')
        PrintMatrix(test_B, 'test_B')
        PrintMatrix(matrix_a_inv, 'test_A_inv')
        PrintMatrix(matrix_b_inv, 'test_B_inv')

        print(f"====================")
        print(f"MatrixMulScalar test:")

        test_A = [[1, 2, 3],
                [4, 5, 6],
                [2, 6, 6]]

        (matrix_c) = MatrixMulScalar(test_A, 20)
        PrintMatrix(test_A, 'test_A')
        PrintMatrix(matrix_c, 'matrix_c')

        print(f"====================")
        print(f"LU Decomposition test:")

        test_A = [[1, 2, 3, 50],
                [2, 1, 7, 100],
                [2, 6, 6,100],
                [4, 55, 24, 12]]

        test_B = [[1, 3, 99, 2, 55],
                [7, 9, 1, 66, 7],
                [4, 88, 21, 76, 333],
                [4, 55, 23, 55, 666],
                [43, 5, 12, 3,  4]]

        (matrix_a_l, matrix_a_u) = LUDecomposition(test_A)
        (matrix_b_l, matrix_b_u) = LUDecomposition(test_B)
        PrintMatrix(test_A, 'test_A')
        PrintMatrix(test_B, 'test_B')
        PrintMatrix(matrix_a_l, 'matrix_a_l')
        PrintMatrix(matrix_a_u, 'matrix_a_u')
        PrintMatrix(matrix_b_l, 'matrix_b_l')
        PrintMatrix(matrix_b_u, 'matrix_b_u')
        print(f"====================")
        print(f"DeterminationCal test:")

        test_A = [[-2, 2, -3],
                [-1, 1, 3],
                [2, 0, -1]]

        test_B = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13,14, 15, 16]]

        test_C = [[1, 2, 3, 4],
                [8, 5, 6, 7],
                [9, 12, 10, 11],
                [13,14, 16, 15]]

        test_D = [[1, 2, 3, 4, 1],
                [8,  5,  6,  7, 2],
                [9, 12, 10, 11, 3],
                [13,14, 16, 15, 4],
                [10, 8,  6,  4, 2]]

        (det_a) = DeterminationCal(test_A)
        (det_b) = DeterminationCal(test_B)
        (det_c) = DeterminationCal(test_C)
        (det_d) = DeterminationCal(test_D)
        PrintMatrix(test_A, 'test_A')
        PrintMatrix(test_B, 'test_B')
        PrintMatrix(test_C, 'test_C')
        PrintMatrix(test_D, 'test_D')
        print(f'det_a = {det_a}')
        print(f'det_b = {det_b}')
        print(f'det_c = {det_c}')
        print(f'det_d = {det_d}')

        print(f"====================")
        PrintMatrix(lse_parameter_x, "lse_parameter_x")

        print(f"====================")
        PrintMatrix(error_result_lse, "error_result_lse")
        print(f"error_val_lse = {error_val_lse}")

        print(f"====================")
        PrintMatrix(newton_parameter_x, "newton_parameter_x")

        print(f"====================")
        PrintMatrix(error_result_newton, "error_result_newton")
        print(f"error_val_newton = {error_val_newton}")

#########################
#     Sub-Routine       #
#########################
def ArgumentParser():
    input_file      = None
    poly_num        = None
    lamb            = None
    is_debug        = 0
    EPSILON         = pow(10, -8)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", "-in_f", help="should set the input file for data.")
    parser.add_argument("--poly_num", "-pln", help="set the number of the polynomial fitting model bases.")
    parser.add_argument("--lamb", "-lmb", help="set the value of lambda for regularization on LSE.")
    parser.add_argument("--epsilon", "-eps", help="set the error precision of the neighbored steps for terminating the Newton's Method. Default 1e^-8.")
    parser.add_argument("--is_debug", "-isd", help="1 for debug mode; 0 for normal mode.")

    args = parser.parse_args()

    if args.input_file:
        input_file = args.input_file
    if args.poly_num:
        poly_num = int(args.poly_num)
    if args.lamb:
        lamb = float(args.lamb)
    if args.epsilon:
        EPSILON = float(args.epsilon)
    if args.is_debug:
        is_debug = int(args.is_debug)

    if(input_file ==  None):
        print(f"Error: You should set input file name with argument '--input_file' or '-in_f'")
        sys.exit()

    if(poly_num ==  None):
        print(f"You did not set the number of the polynomial fitting model bases.")
        print(f"It will be set to 0.")
        poly_num = 0

    if(lamb ==  None):
        print(f"You did not set the value of lambda for regularization on LSE.")
        print(f"It will be set to 0.")
        lamb = 0

    return (input_file, poly_num, lamb, is_debug, EPSILON)

def Visualization(lse_parameter_x, newton_parameter_x, input_data, error_val_lse, error_val_newton, is_debug):
    #creat the subplot object
    fig=plt.subplots(1,2)

    #======================LSE========================#
    plt.subplot(2, 1, 1)
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
    plt.title("LSE Method")
    plt.scatter(x, y, c='red', s=15, zorder=3)

    #plot the fitting line
    fitted_str = GenFittedString([rows[0] for rows in lse_parameter_x])
    print(f'LSE: ')
    print(f'Fitting line: {fitted_str}')
    print(f'Total error: {error_val_lse}')
    fitted = np.poly1d([rows[0] for rows in lse_parameter_x])
    xaxis_range = np.arange(-7, 7, 0.01)
    yaxis_range = fitted(xaxis_range)
    plt.plot(xaxis_range, yaxis_range)
    if(is_debug):
        print(f"LSE, xaxis_range = {xaxis_range}")
        print(f"LSE, yaxis_range = {yaxis_range}")
        print(f"================================")

    #======================Newton========================#
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
    fitted_str = GenFittedString([rows[0] for rows in newton_parameter_x])
    print(f"Newton's Method: ")
    print(f'Fitting line: {fitted_str}')
    print(f'Total error: {error_val_newton}')
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

def GenFittedString(parameter_x):
    fitted_string = ''
    len_parameter = len(parameter_x)
    for x in range(len_parameter):
        if(x==len_parameter-1):
            fitted_string += str(abs(parameter_x[x]))
        else:
            if(parameter_x[x+1]>=0):
                if(x==0):
                    fitted_string += str(parameter_x[x])+'X^'+str(len_parameter-1-x)+" + "
                else:
                    fitted_string += str(abs(parameter_x[x]))+'X^'+str(len_parameter-1-x)+" + "
            else:
                if(x==0):
                    fitted_string += str(parameter_x[x])+'X^'+str(len_parameter-1-x)+" - "
                else:
                    fitted_string += str(abs(parameter_x[x]))+'X^'+str(len_parameter-1-x)+" - "

    return fitted_string

def ErrorCalculation(design_matrix, parameter_x, b):
    matrix_axminb   = MatrixSub(MatrixMul(design_matrix, parameter_x), b) #A*x-b
    matrix_axminb_t = MatrixTranspose(matrix_axminb) #(A*x-b)^T
    error_result    = MatrixMul(matrix_axminb_t, matrix_axminb) #||A*x-b||^2

    return (error_result[0][0], error_result)

def ErrorVectorCalculation(vector_x):
    return (math.sqrt(MatrixMul(MatrixTranspose(vector_x), vector_x)[0][0])) # sqrt(||x||^2)

def NewtonMethod(design_matrix, b, is_debug, EPSILON):
    parameter_x               = MatrixTranspose([[0 for i in range(len(design_matrix[0]))]]) #x0 initial point
    design_matrix_t           = MatrixTranspose(design_matrix)
    tmp_matrix_2atb           = MatrixMulScalar(MatrixMul(design_matrix_t, b), 2) #2AT*b
    tmp_design_matrix_gram    = MatrixMul(design_matrix_t, design_matrix) #AT*A
    tmp_design_matrix_2gram   = MatrixMulScalar(tmp_design_matrix_gram, 2) #2(AT*A)
    hessian_matrix            = tmp_design_matrix_2gram #2(AT*A)
    hessian_matrix_inv        = MatrixInverse(hessian_matrix)
    index = 0

    while (True):
        tmp_design_matrix_2gram_x = MatrixMul(tmp_design_matrix_2gram, parameter_x) #2AT*A*x
        matrix_gradient_x = MatrixSub(tmp_design_matrix_2gram_x, tmp_matrix_2atb) #2AT*A*x-2AT*b
        tmp_parameter_x   = MatrixSub(parameter_x, MatrixMul(hessian_matrix_inv, matrix_gradient_x)) #x1 = x0 - ((H)^-1)*(gradient_x)
        error_try         = ErrorVectorCalculation(MatrixSub(tmp_parameter_x, parameter_x))

        if(error_try < EPSILON):
            parameter_x = tmp_parameter_x
            break
        else:
            if(is_debug):
                print(f"=====index = {index}=====")
                PrintMatrix(parameter_x, "newton_parameter_x")
                PrintMatrix(tmp_parameter_x, "tmp_newton_parameter_x")
                PrintMatrix(MatrixSub(tmp_parameter_x, parameter_x), "MatrixSub()")
                print(f"error_try = {error_try}")
                print(f"EPSILON = {EPSILON}")

        parameter_x = tmp_parameter_x
        index += 1
    return parameter_x

def LSEMethod(design_matrix, identity_matrix, b):
    design_matrix_t = MatrixTranspose(design_matrix)
    tmp_matrix      = MatrixAdd(MatrixMul(design_matrix_t, design_matrix), identity_matrix) #AT*A+lamb*I
    tmp_matrix_inv  = MatrixInverse(tmp_matrix) #(AT*A+lamb*I)^-1
    tmp_atb         = MatrixMul(design_matrix_t, b) #AT*b
    parameter_x     = MatrixMul(tmp_matrix_inv, tmp_atb)#((AT*A+lamb*I)^-1)*AT*b

    return parameter_x

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

def PrintMatrix(input_matrix, matrix_name):
    print(f'{matrix_name} = ')
    print(f'[', end = '')
    for index_i, rows in enumerate(input_matrix):
        for index_j, cols in enumerate(rows):
            if(index_i == (len(input_matrix)-1) and index_j == (len(rows)-1)):
                print(f'{input_matrix[index_i][index_j]}]') #will print the same
                #print(f'[{cols}] ') #will print the same
            elif(index_j == (len(rows)-1)):
                print(f'{input_matrix[index_i][index_j]} ') #will print the same
            else:
                if(index_j == 0 and index_i != 0):
                    print(f' {input_matrix[index_i][index_j]} ', end='') #will print the same
                else:
                    print(f'{input_matrix[index_i][index_j]} ', end='') #will print the same


def FormMatrix(input_data, poly_num):
    design_matrix = []
    b             = []

    for data_pairs in input_data:
        row_val = []
        for index in range(poly_num):
            row_val.insert(0, pow(data_pairs[0], index))

        design_matrix.append(row_val)
        b.append([data_pairs[1]])

    return(design_matrix, b)

def ReadInputFile(input_file):
    file_data  = open(input_file, 'r')
    file_lines = file_data.readlines()
    input_data = []

    for line in file_lines:
        m = re.match(r"\s*(\S+)\s*\,\s*(\S+)\s*", line)
        if(m is not None):
            data_pairs = []
            data_pairs.append(float(m.group(1)))
            data_pairs.append(float(m.group(2)))
            input_data.append(data_pairs)

    return input_data

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
