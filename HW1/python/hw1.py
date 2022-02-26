#! /usr/bin/env python3
'''
    Author      : BCC
    Date        : 2022/02/24
'''

import argparse
import math
import sys
import re
import numpy as np

EPSILON = pow(10, -14) #wiki 1e^-14

#########################
#     Main-Routine      #
#########################
def main():
    #Process the argument
    (input_file, poly_num, lamb, is_debug) = ArgumentParser()

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
    newton_parameter_x = NewtonMethod(design_matrix, b, is_debug)

    #CalculateError for Method2
    (error_val_newton, error_result_newton) = ErrorCalculation(design_matrix, newton_parameter_x, b)

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
                [2, 6, 6]]

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", "-in_f", help="should set the input file for data.")
    parser.add_argument("--poly_num", "-pln", help="set the number of the polynomial fitting model bases.")
    parser.add_argument("--lamb", "-lmb", help="set the value of lambda for regularization on LSE.")
    parser.add_argument("--is_debug", "-isd", help="1 for debug mode; 0 for normal mode.")

    args = parser.parse_args()

    if args.input_file:
        input_file = args.input_file
    if args.poly_num:
        poly_num = int(args.poly_num)
    if args.lamb:
        lamb = float(args.lamb)
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

    return (input_file, poly_num, lamb, is_debug)

def ErrorCalculation(design_matrix, parameter_x, b):
    matrix_axminb   = MatrixSub(MatrixMul(design_matrix, parameter_x), b) #A*x-b
    matrix_axminb_t = MatrixTranspose(matrix_axminb) #(A*x-b)^T
    error_result    = MatrixMul(matrix_axminb_t, matrix_axminb) #||A*x-b||^2

    return (error_result[0][0], error_result)

def ErrorVectorCalculation(vector_x):
    return (math.sqrt(MatrixMul(MatrixTranspose(vector_x), vector_x)[0][0])) # sqrt(||x||^2)

def NewtonMethod(design_matrix, b, is_debug):
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

    #initialization matrix_c
    for i in range(row_a):
        row = []
        for j in range(col_a):
            row.append(0)

        matrix_c.append(row)

    #multiply every element with the scalar
    for i in range(row_a):
        for j in range(col_a):
            matrix_c[i][j] = scalar*matrix_a[i][j]

    return matrix_c

def MatrixInverse(matrix_a):
    matrix_c = np.linalg.inv(np.array(matrix_a))
    matrix_c.tolist()

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

    #initialization matrix_c
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

    #initialization matrix_c
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

    #initialization matrix_c
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

    #initialization matrix_c
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

def LUDecomposition():
    a = 1
    return ()

#---------------Execution---------------#
if __name__ == '__main__':
    main()
