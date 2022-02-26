#! /usr/bin/env python3
'''
    Author      : BCC
    Date        : 2022/02/24

'''

import argparse
import sys
import re
#import numpy as np

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


    test_A = [[1, 2, 3],
              [4, 5, 6]]

    test_B = [[1, 3],
              [7, 9],
              [13, 0]]

    (matrix_c) = MatrixMul(test_A, test_B)

    PrintMatrix(test_A, 'test_A')
    PrintMatrix(test_B, 'test_B')
    PrintMatrix(matrix_c, 'matrix_c')



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
        lamb = args.lamb
    if args.is_debug:
        is_debug = int(args.is_debug)

    if(input_file ==  None):
        print(f"You should set input file name with argument '--input_file' or '-in_f'")
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

def MatrixMul(matrix_a, matrix_b):
    row_a = len(matrix_a)
    col_a = len(matrix_a[0])
    row_b = len(matrix_b)
    col_b = len(matrix_b[0])

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
