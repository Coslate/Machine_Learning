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
    (input_training_dir, input_testing_dir, largest_k, output_dir, is_debug) = ArgumentParser()

    print(f"> ReadInputFile...")
    train_img_data, train_img_label, row, col = ReadInputFile(input_training_dir)
    test_img_data, test_img_label, row, col   = ReadInputFile(input_testing_dir)

    print(f"> PerformPCA...")
    PerformPCA(train_img_data, train_img_label, test_img_data, test_img_label, largest_k, row, col, output_dir)

    if(is_debug):
        #pass
        print(f"len(train_img_data) = {len(train_img_data)}")
        print(f"train_img_data[0].shape = {train_img_data[0].shape}")
        print(f"len(test_img_data) = {len(test_img_data)}")
        print(f"test_img_data[0].shape = {test_img_data[0].shape}")
        #img1 = Image.fromarray(train_img_data[0])
        #img2 = Image.fromarray(test_img_data[0])
        #plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
        #plt.show()
        #img1.save(f'{output_dir}/train_img0.png')
        #img2.save(f'{output_dir}/test_img0.png')

#########################
#     Sub-Routine       #
#########################
def ArgumentParser():
    input_training_dir  = None
    input_testing_dir   = None
    largest_k           = 30
    output_dir          = "./output"
    is_debug            = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_training_dir",    "-itrd",     help="The directory name includes all training images.")
    parser.add_argument("--input_testing_dir",     "-ited",     help="The directory name includes all testing images.")
    parser.add_argument("--largest_k",            "-lmk",       help="The number of largest k eigenvectors that will be chosen in PCA. Default is 30.")
    parser.add_argument("--output_dir",            "-odr",      help="The output directory of the result. Default is './output'")
    parser.add_argument("--is_debug",              "-isd",      help="1 for debug mode; 0 for normal mode.")

    args = parser.parse_args()

    if(args.input_training_dir):
        input_training_dir = args.input_training_dir
    if(args.input_testing_dir):
        input_testing_dir  = args.input_testing_dir
    if(args.largest_k):
        largest_k         = int(args.largest_k)
    if(args.output_dir):
        output_dir         = args.output_dir
    if(args.is_debug):
        is_debug           = int(args.is_debug)

    if(input_training_dir == None):
        print(f"Error: You should set '--input_training_dir' or '-itrd' for the directory name that includes all the training images.")
        sys.exit()

    if(input_testing_dir == None):
        print(f"Error: You should set '--input_testing_dir' or '-ited' for the directory name that includes all the testing images.")
        sys.exit()

    if(is_debug):
        print(f"input_training_dir   = {input_training_dir}")
        print(f"input_testing_dir    = {input_testing_dir}")
        print(f"largest_k   = {largest_k}")
        print(f"output_dir   = {output_dir}")
        print(f"is_debug     = {is_debug}")

    return (input_training_dir, input_testing_dir, largest_k, output_dir, is_debug)

def FormCovMatrix(data_matrix):
    N          = data_matrix.shape[0] #135
    orig_dimen = data_matrix.shape[1] #1189
    data_mean  = np.zeros(orig_dimen, dtype=np.float64) #1x1189
    x_min_xmean= np.zeros((data_matrix.shape[1], data_matrix.shape[0]), dtype=np.float64) #x-ux

    #Calculate the mean
    for i in range(data_matrix.shape[0]):
        for j in range(data_matrix.shape[1]):
            data_mean[j] += data_matrix[i][j]
    data_mean = data_mean/N

    #Calculate x-ux
    for i in range(data_matrix.shape[0]):
        for j in range(data_matrix.shape[1]):
            x_min_xmean[j][i] = data_matrix[i][j] - data_mean[j]

    cov_matrix = x_min_xmean @ x_min_xmean.T
    cov_matrix = cov_matrix/N

    return cov_matrix

def FormMaxKEigenMatrix(in_matrix, total_n, largest_k):
    eigen_values, eigen_vectors = np.linalg.eig(in_matrix)
    #eigen_matrix = np.zeros((total_n, largest_k), dtype=np.float64)
    eigen_matrix = np.zeros((total_n, largest_k), dtype=eigen_vectors.dtype)

    #Select k max eigen_vectors
    eigen_values_chosen = eigen_values.copy()
    has_chosen = 0
    while(has_chosen < largest_k):
        chosen_idx = np.argmax(eigen_values_chosen)

        #fill in the largest eigen_vectors in eigen_matrix
        for i in range(eigen_matrix.shape[0]): #foreach row
#            eigen_matrix[i][has_chosen] = eigen_vectors[i][chosen_idx].real
            eigen_matrix[i][has_chosen] = eigen_vectors[i][chosen_idx]

        eigen_values_chosen[chosen_idx] = -np.inf
        has_chosen += 1
    return eigen_matrix

def ShowEigenFaces(eigen_mat, row, col, directory):
    total_num       = eigen_mat.shape[1]
    eigen_mat_trans = eigen_mat.T

    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        #clean all gif and png files
        for zippath in glob.iglob(os.path.join(directory, '*eigen*.png')):
            os.remove(zippath)

    for index in range(total_num):
        eigen_face = eigen_mat_trans[index].reshape(row, col)
        #scale to 0-255
        eigen_face = ((eigen_face - eigen_face.min()) * (1/(eigen_face.max() - eigen_face.min()) * 255)).astype('uint8')
        out_file_name = directory + "/pca_eigen"+str(index+1)+".png"
        img_data_gen = eigen_face.astype(np.uint8)
        img_data_gen = Image.fromarray(img_data_gen)
        img_data_gen.save(out_file_name)

def FacesReconstruction(train_img_data, eigen_mat, directory, row, col):
    orig_dim   = train_img_data.shape[1]
    new_dim    = eigen_mat.shape[1]
    picked_num = np.random.choice(train_img_data.shape[0], 10, replace=False)
    for i in range(10):
        picked_x      = train_img_data[picked_num[i]]
        reconstruct_z = picked_x.reshape(1, orig_dim) @ eigen_mat @ eigen_mat.T
        reconstruct_z = reconstruct_z.reshape(row, col)
        reconstruct_z = ((reconstruct_z - reconstruct_z.min()) * (1/(reconstruct_z.max() - reconstruct_z.min()) * 255)).astype('uint8')

        out_file_origin_name = directory + "/pca_origin"+str(i+1)+".png"
        img_data_gen = picked_x.reshape(row, col).astype(np.uint8)
        img_data_gen = Image.fromarray(img_data_gen)
        img_data_gen.save(out_file_origin_name)
        out_file_recon_name = directory + "/pca_reconstruct"+str(i+1)+".png"
        img_data_gen = reconstruct_z.astype(np.uint8)
        img_data_gen = Image.fromarray(img_data_gen)
        img_data_gen.save(out_file_recon_name)

def PerformPCA(train_img_data, train_img_label, test_img_data, test_img_label, largest_k, row, col, output_dir):
    output_dir += "/PCA"

    cov_mat    = FormCovMatrix(train_img_data)
    eigen_mat  = FormMaxKEigenMatrix(cov_mat, cov_mat.shape[0], largest_k)
    ShowEigenFaces(eigen_mat, row, col, output_dir)
    FacesReconstruction(train_img_data, eigen_mat, output_dir, row, col)
    print(f"eigen_mat.shape = {eigen_mat.shape}")
    print(f"eigen_mat = {eigen_mat}")


def ReadInputFile(input_directory):
    #Get the list of image files under the directory
    img_filename = []
    row = 100
    col = 100
#    row = 29
#    col = 41
#    row = 29
#    col = 24

    for file in glob.glob(input_directory+"/*.pgm"):
        img_filename.append(file)

    img_data_array  = np.zeros((len(img_filename), row*col), dtype=np.uint8)
    img_label_array = np.zeros(len(img_filename), dtype=int)

    for i, file_name in enumerate(img_filename):
        im     = Image.open(file_name)
        new_im = im.resize((col, row))
        im_data = np.array(new_im)
        img_data_array[i]  = im_data.reshape(1, -1)
        match_result = re.match(r'.*\/subject(\d+)\..*', file_name)
        img_label_array[i] = match_result.group(1)

    return img_data_array, img_label_array, row, col


#---------------Execution---------------#
if __name__ == '__main__':
    main()
