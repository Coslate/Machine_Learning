#! /usr/bin/env python3
'''
    Author      : BCC
    Date        : 2022/05/02
'''

import argparse
import math
import sys
import re
import copy
import numpy as np
import matplotlib.pyplot as plt
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
    (input_img1, input_img2, is_debug) = ArgumentParser()

    print(f"> ReadInputFile...")
    img_data1 = ReadInputFile(input_img1)
    img_data2 = ReadInputFile(input_img2)

    if(is_debug):
        print(f"im_data1.type = {img_data1.shape}")
        print(f"im_data2.type = {img_data2.shape}")
        img1 = Image.fromarray(img_data1)
        img2 = Image.fromarray(img_data2)
        img1.save(r'./data/test_img1.png')
        img2.save(r'./data/test_img2.png')
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(img1)
        ax[1].imshow(img2)
        plt.show()
#       display(img1)
#       display(img2)

#########################
#     Sub-Routine       #
#########################
def ArgumentParser():
    input_img1          = None
    input_img2          = None
    is_debug            = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_img1",   "-img1",     help="The file name of the first input image.")
    parser.add_argument("--input_img2",   "-img2",     help="The file name of the second input image.")
    parser.add_argument("--is_debug",    "-isd",      help="1 for debug mode; 0 for normal mode.")

    args = parser.parse_args()

    if(args.input_img1):
        input_img1   = args.input_img1
    if(args.input_img2):
        input_img2   = args.input_img2
    if(args.is_debug):
        is_debug     = int(args.is_debug)

    if(input_img1 == None):
        print(f"Error: You should set '--input_img1' or '-img1' for the file name of the first input image.")
        sys.exit()

    if(input_img2 == None):
        print(f"Error: You should set '--input_img2' or '-img2' for the file name of the second input image.")
        sys.exit()

    if(is_debug):
        print(f"input_img1   = {input_img1}")
        print(f"input_img2   = {input_img2}")
        print(f"is_debug     = {is_debug}")

    return (input_img1, input_img2, is_debug)

def ReadInputFile(input_file):
    im      = Image.open(input_file)
#    im_data = np.array(im.getdata())
    im_data = np.array(im)
    return im_data

#---------------Execution---------------#
if __name__ == '__main__':
    main()
