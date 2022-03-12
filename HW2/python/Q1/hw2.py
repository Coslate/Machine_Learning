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
    (infile_train_label, infile_train_image, infile_test_label, infile_test_image, toggle, is_debug) = ArgumentParser()

    #Get the input data point
    train_y = ReadLabelFile(infile_train_label, 2049, is_debug)
    train_x = ReadImageFile(infile_train_image, 2051, is_debug)
    test_y  = ReadLabelFile(infile_test_label, 2049, is_debug)
    test_x  = ReadImageFile(infile_test_image, 2051, is_debug)

    #Print the debug messages when necessary
    if(is_debug):
        print(f"infile_train_label = {infile_train_label}")
        print(f"infile_train_image = {infile_train_image}")
        print(f"infile_test_label  = {infile_test_label}")
        print(f"infile_test_image  = {infile_test_image}")
        print(f"toggle = {toggle}")
        i = 50145
        print(f"train_y[{i}] = {train_y[i]}")
        ShowImage(train_x[i])

        i = 58943
        print(f"train_y[{i}] = {train_y[i]}")
        ShowImage(train_x[i])

        i = 33134
        print(f"train_y[{i}] = {train_y[i]}")
        ShowImage(train_x[i])

        i = 36158
        print(f"train_y[{i}] = {train_y[i]}")
        ShowImage(train_x[i])

        i = 8191
        print(f"test_y[{i}] = {test_y[i]}")
        ShowImage(test_x[i])

        i = 548
        print(f"test_y[{i}] = {test_y[i]}")
        ShowImage(test_x[i])

        i = 777
        print(f"test_y[{i}] = {test_y[i]}")
        ShowImage(test_x[i])

        pass


#########################
#     Sub-Routine       #
#########################
def ArgumentParser():
    infile_train_label= None
    infile_train_image= None
    infile_test_label = None
    infile_test_image = None
    toggle            = None
    is_debug          = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--infile_train_label", "-inf_tr_label", help="should set the input training label file.")
    parser.add_argument("--infile_train_image", "-inf_tr_image", help="should set the input training image file.")
    parser.add_argument("--infile_test_label", "-inf_tst_label", help="should set the input testing label file.")
    parser.add_argument("--infile_test_image", "-inf_tst_image", help="should set the input testing image file.")
    parser.add_argument("--toggle", "-tgl", help="set '0' for discrete mode, '1' for continuous mode.")
    parser.add_argument("--is_debug", "-isd", help="1 for debug mode; 0 for normal mode.")

    args = parser.parse_args()

    if args.infile_train_label:
        infile_train_label = args.infile_train_label
    if args.infile_train_image:
        infile_train_image = args.infile_train_image
    if args.infile_test_label:
        infile_test_label = args.infile_test_label
    if args.infile_test_image:
        infile_test_image = args.infile_test_image
    if args.toggle:
        toggle = int(args.toggle)
    if args.is_debug:
        is_debug = int(args.is_debug)

    if(infile_train_label ==  None):
        print(f"Error: You should set input file name with for training label '--infile_train_label' or '-inf_tr_label'")
        sys.exit()

    if(infile_train_label ==  None):
        print(f"Error: You should set input file name with for training image '--infile_train_label' or '-inf_tr_image'")
        sys.exit()

    if(infile_test_label ==  None):
        print(f"Error: You should set input file name with for testing label '--infile_test_label' or '-inf_tst_label'")
        sys.exit()

    if(infile_test_image ==  None):
        print(f"Error: You should set input file name with for testing image '--infile_test_image' or '-inf_tst_image'")
        sys.exit()

    if(toggle ==  None):
        print(f"Warning: You did not set the value of toggle.")
        print(f"Warning: It will be set to 0.")
        toggle = 0

    return (infile_train_label, infile_train_image, infile_test_label, infile_test_image, toggle, is_debug)


def ReadLabelFile(file_name, magic_test_num, is_debug):
    magic_num    = None
    num_of_items = None
    y_data       = []
    m = re.match(r"\s*(\S+)\.gz", file_name)
    if(m is not None):
        file_in = gzip.open(file_name, 'rb')
    else:
        file_in = open(file_name, "rb")

    magic_num = int.from_bytes(file_in.read(4), 'big')
    if(magic_num != magic_test_num):
        print(f"Error: Your magic_num:{magic_num} of file:{file_name} does not match to an expected magic number : {magic_test_num}.")
        sys.exit()

    num_of_items = int.from_bytes(file_in.read(4), 'big')
    if(num_of_items is None):
        print(f"Error: The num_of_items is None. Not expected. The format of the file:{file_name} is not expected.")
        sys.exit()

    while(True):
        label_data = file_in.read(1)
        if(label_data):
            label_data = int.from_bytes(label_data, 'big')
            y_data.append(label_data)
        else:
            break

    if(is_debug):
        print(f"file         = {file_name}")
        print(f"magic_num    = {magic_num}")
        print(f"num_of_items = {num_of_items}")

    return y_data

def ReadImageFile(file_name, magic_test_num, is_debug):
    magic_num    = None
    num_of_items = None
    num_of_rows  = None
    num_of_cols  = None
    x_data       = []
    image_data   = []

    m = re.match(r"\s*(\S+)\.gz", file_name)
    if(m is not None):
        file_in = gzip.open(file_name, 'rb')
    else:
        file_in = open(file_name, "rb")

    magic_num = int.from_bytes(file_in.read(4), 'big')
    if(magic_num != magic_test_num):
        print(f"Error: Your magic_num:{magic_num} of file:{file_name} does not match to an expected magic number : {magic_test_num}.")
        sys.exit()

    num_of_items = int.from_bytes(file_in.read(4), 'big')
    if(num_of_items is None):
        print(f"Error: The num_of_items is None. Not expected. The format of the file:{file_name} is not expected.")
        sys.exit()

    num_of_rows = int.from_bytes(file_in.read(4), 'big')
    if(num_of_rows is None):
        print(f"Error: The num_of_rows is None. Not expected. The format of the file:{file_name} is not expected.")
        sys.exit()

    num_of_cols = int.from_bytes(file_in.read(4), 'big')
    if(num_of_cols is None):
        print(f"Error: The num_of_cols is None. Not expected. The format of the file:{file_name} is not expected.")
        sys.exit()

    image_data = file_in.read()
    if(image_data):
        row          = []
        image_matrix = []
        index_i      = 0
        index_j      = 0
        for b in image_data:
            row.append(b)
            index_j += 1

            if(index_j%num_of_cols == 0):
                image_matrix.append(row)
                row     = []
                index_i +=1
                index_j = 0
            if(index_i%num_of_rows == 0 and index_i != 0):
                index_i = 0
                x_data.append(image_matrix)
                image_matrix = []

    if(is_debug):
        print(f"file         = {file_name}")
        print(f"magic_num    = {magic_num}")
        print(f"num_of_items = {num_of_items}")
        print(f"num_of_rows  = {num_of_rows}")
        print(f"num_of_cols  = {num_of_cols}")
        print(f"len(x_data) = {len(x_data)}, x_data = ")
#        for i in range(len(x_data)):
#            print(f"i = {i}, {x_data[i]}")

    return x_data

def ShowImage(image_matrix):
    im = plt.imshow(image_matrix, cmap='gray', vmin=0, vmax=255)
    plt.show()




#---------------Execution---------------#
if __name__ == '__main__':
    main()
