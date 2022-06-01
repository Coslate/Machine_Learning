#! /usr/bin/env python3
'''
    Author      : BCC
    Date        : 2022/06/01
'''

import argparse
import math
import sys
import re
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import numba as nb
from scipy.spatial import distance
from PIL import Image
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
    print(f"> ArgumentParser...")
    (input_training_dir, input_testing_dir, epsilon_err, output_dir, is_debug) = ArgumentParser()

    print(f"> ReadInputFile...")
    train_img_data = ReadInputFile(input_training_dir)
    test_img_data  = ReadInputFile(input_testing_dir)

    if(is_debug):
        print(f"len(train_img_data) = {len(train_img_data)}")
        print(f"train_img_data[0].shape = {train_img_data[0].shape}")
        print(f"len(test_img_data) = {len(test_img_data)}")
        print(f"test_img_data[0].shape = {test_img_data[0].shape}")
        pass
#        img1 = Image.fromarray(train_img_data[0])
#        plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
#        plt.show()
#        img1.save(f'{output_dir}/test_img1.png')

#########################
#     Sub-Routine       #
#########################
def ArgumentParser():
    input_training_dir  = None
    input_testing_dir   = None
    epsilon_err         = 0.0001
    output_dir          = "./output"
    is_debug            = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_training_dir",    "-itrd",     help="The directory name includes all training images.")
    parser.add_argument("--input_testing_dir",     "-ited",     help="The directory name includes all testing images.")
    parser.add_argument("--epsilon_err",           "-eps",      help="The error in neighbored step that Kmeans can terminate. Default is 0.0001.")
    parser.add_argument("--output_dir",            "-odr",      help="The output directory of the result. Default is './output'")
    parser.add_argument("--is_debug",              "-isd",      help="1 for debug mode; 0 for normal mode.")

    args = parser.parse_args()

    if(args.input_training_dir):
        input_training_dir   = args.input_training_dir
    if(args.input_testing_dir):
        input_testing_dir   = args.input_testing_dir
    if(args.epsilon_err):
        epsilon_err  = float(args.epsilon_err)
    if(args.output_dir):
        output_dir         = args.output_dir
    if(args.is_debug):
        is_debug          = int(args.is_debug)

    if(input_training_dir == None):
        print(f"Error: You should set '--input_training_dir' or '-itrd' for the directory name that includes all the training images.")
        sys.exit()

    if(input_testing_dir == None):
        print(f"Error: You should set '--input_testing_dir' or '-ited' for the directory name that includes all the testing images.")
        sys.exit()

    if(is_debug):
        print(f"input_training_dir   = {input_training_dir}")
        print(f"input_testing_dir    = {input_testing_dir}")
        print(f"epsilon_err  = {epsilon_err}")
        print(f"output_dir   = {output_dir}")
        print(f"is_debug     = {is_debug}")

    return (input_training_dir, input_testing_dir, epsilon_err, output_dir, is_debug)

def ReadInputFile(input_directory):
    #Get the list of image files under the directory
    img_filename = []
    row = 29
    col = 41

    for file in glob.glob(input_directory+"/*.pgm"):
        img_filename.append(file)

    img_data_array = np.zeros((len(img_filename), row, col), dtype=np.uint8)

    for i, file_name in enumerate(img_filename):
        im     = Image.open(file_name)
        new_im = im.resize((col, row))
        im_data = np.array(new_im)
        img_data_array[i] = im_data

#        print(f"file_name = {file_name}")
#        print(f"type(im_data) = {type(im_data)}")
#        print(f"im_data.shape = {im_data.shape}")
#        print(f"im_data = {im_data}")
#        img1 = Image.fromarray(im_data)
#        plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
#        plt.show()
#        a = input()

    return img_data_array


#---------------Execution---------------#
if __name__ == '__main__':
    main()
