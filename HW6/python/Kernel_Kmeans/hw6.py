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
    (input_img1, input_img2, gs, gc, init_method, cluster_num, epsilon_err, is_debug) = ArgumentParser()

    print(f"> ReadInputFile...")
    img_data1 = ReadInputFile(input_img1)
    img_data2 = ReadInputFile(input_img2)

    print(f"> KernelKmeans...")
    (cluster_img1_list) = KernelKmeans(img_data1, gs, gc, init_method, cluster_num, epsilon_err)
    (cluster_img2_list) = KernelKmeans(img_data2, gs, gc, init_method, cluster_num, epsilon_err)

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
    gs                  = 0.0001
    gc                  = 0.0001
    init_method         = 0
    cluster_num         = 2
    epsilon_err         = 0.0001
    is_debug            = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_img1",   "-img1",     help="The file name of the first input image.")
    parser.add_argument("--input_img2",   "-img2",     help="The file name of the second input image.")
    parser.add_argument("--gs",           "-gs",       help="The gamma hyper parameter for RBF kernel of space.")
    parser.add_argument("--gc",           "-gc",       help="The gamma hyper parameter for RBF kernel of color.")
    parser.add_argument("--init_method",  "-init_m",   help="The method for choosing initial centroids for clustering. Set 0 for random selection. Set 1 for kmeans++ selection. Default 0.")
    parser.add_argument("--cluster_num",  "-cn",       help="The number of clusters you want. Default is 2.")
    parser.add_argument("--epsilon_err",  "-eps",      help="The error in neighbored step that Kmeans can terminate. Default is 0.0001.")
    parser.add_argument("--is_debug",    "-isd",       help="1 for debug mode; 0 for normal mode.")

    args = parser.parse_args()

    if(args.input_img1):
        input_img1   = args.input_img1
    if(args.input_img2):
        input_img2   = args.input_img2
    if(args.gs):
        gs           = float(args.gs)
    if(args.gc):
        gc           = float(args.gc)
    if(args.init_method):
        init_method  = int(args.init_method)
    if(args.cluster_num):
        cluster_num  = int(args.cluster_num)
    if(args.epsilon_err):
        epsilon_err  = float(args.epsilon_err)
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
        print(f"gs           = {gs}")
        print(f"gc           = {gc}")
        print(f"init_method  = {init_method}")
        print(f"cluster_num  = {cluster_num}")
        print(f"epsilon_err  = {epsilon_err}")
        print(f"is_debug     = {is_debug}")

    return (input_img1, input_img2, gs, gc, init_method, cluster_num, epsilon_err, is_debug)

def CalculateKernelFunctions(img_data, gs, gc, row, col, color_dim):
    kernel                = np.zeros((row*col, row*col), dtype = np.float64)

    #Calculate the phi_color gram matrix
    phi_color             = np.reshape(img_data, (row*col, color_dim))
    color_dist_matrix     = distance.cdist(phi_color, phi_color, 'sqeuclidean')
    color_dist_matrix     = np.exp(-1*gc*color_dist_matrix)

    #Calculate the phi_space gram matrix
    phi_space = []
    for i in range(row):
        for j in range(col):
            phi_space.append([i, j])
    phi_space = np.array(phi_space)
    space_dist_matrix     = distance.cdist(phi_space, phi_space, 'sqeuclidean')
    space_dist_matrix     = np.exp(-1*gs*space_dist_matrix)

    #Combine to construct the kernel, element-wised matrix multiplication
    kernel = color_dist_matrix*space_dist_matrix

    return kernel

def CenterInitialization(init_method, cluster_num, row, col):
    cluster_result = np.full(row*col, -1, dtype=np.float64) #10000x1
    init_centroids = np.zeros((cluster_num, 2), dtype=np.float64) #kx2

    #Select K points to be the initialized init_centroids
    if(init_method == 0):
        #Random Selection
        coords_x = (np.random.randint(0, row+1, cluster_num)).reshape(-1, 1)
        coords_y = (np.random.randint(0, col+1, cluster_num)).reshape(-1, 1)
        init_centroids = np.hstack((coords_x, coords_y))
    elif(init_method == 1):
        #Kmeans++
        pass

    #Initialize to the center points
    for clus_num, center in enumerate(init_centroids):
        i = center[1]
        j = center[0]
        pos = i*col + j
        cluster_result[pos] = clus_num

    return cluster_result, init_centroids

def CalculateEachClusterGroup(cluster_result, cluster_num):
    cluster_group = [[] for x in range(cluster_num)]
    cluster_size  = np.zeros(cluster_num, dtype=np.float64)

    for pixel_pos, assigned_cluster in enumerate(cluster_result):
        if(assigned_cluster == -1):
            continue

        cluster_size[int(assigned_cluster)] += 1
        cluster_group[int(assigned_cluster)].append(pixel_pos)

    return cluster_size, cluster_group

@nb.jit(nopython=True, nogil=True)
def SumWithOtherClusterMem(pixel_pos, cluster_group, kernel):
    kernel_sum = 0
    for pixel_pos_clusmem in cluster_group:
        kernel_sum += kernel[pixel_pos][pixel_pos_clusmem]

    return kernel_sum

@nb.jit(nopython=True, nogil=True)
def SumOtherTwoClusterMem(cluster_group, kernel):
    cluster_tot_size = len(cluster_group)
    kernel_sum   = 0

    for i in range(cluster_tot_size):
        for j in range(cluster_tot_size):
            kernel_sum += kernel[cluster_group[i]][cluster_group[j]]

    return kernel_sum

@nb.jit(nopython=True, nogil=True)
def UpdateClusterIndicator(new_cluster_result, total_size_image, kernel, init_centroids, cluster_group, cluster_size, cluster_num):
    #Assign each pixel to its closest center
    for pixel_pos in range(total_size_image):
        distance = np.zeros(cluster_num, dtype=np.float64)

        #foreach of the cluster
        for clus_num in range(cluster_num):
            sum_with_other_mem = SumWithOtherClusterMem(pixel_pos, cluster_group[clus_num], kernel)
            sum_other_two_mem  = SumOtherTwoClusterMem(cluster_group[clus_num], kernel)

            distance[clus_num] = kernel[pixel_pos][pixel_pos] - (2.0/cluster_size[clus_num])*sum_with_other_mem + (1/(cluster_size[clus_num]*cluster_size[clus_num]))*sum_other_two_mem

        new_cluster_result[pixel_pos] = np.argmin(distance)

    return new_cluster_result


def KmeansClustering(cluster_result, init_centroids, cluster_num, kernel, row, col):
    print(f">>>> CalculateEachClusterGroup...")
    (cluster_size, cluster_group)  = CalculateEachClusterGroup(cluster_result, cluster_num)
    cluster_group = np.array(cluster_group)

    print(f">>>> Assignment of each pixel to its closest center...")
    #Assign each pixel to its closest center
    new_cluster_result = cluster_result.copy()
    total_size_image   = row*col
    new_cluster_result = UpdateClusterIndicator(new_cluster_result, total_size_image, kernel, init_centroids, cluster_group, cluster_size, cluster_num)

    return new_cluster_result

def KmeansAlg(cluster_result, init_centroids, img_data, row, col, cluster_num, kernel, epsilon_err):
    count          = 1

    print(f">>> KmeansClustering...")
    while(True):
        #Classify all samples according to the closest mean(init_centroids)
        new_cluster_result = KmeansClustering(cluster_result, init_centroids, cluster_num, kernel, row, col)
        error_result       = np.linalg.norm((new_cluster_result - cluster_result), ord=2)

        cluster_result = new_cluster_result.copy()
        if(error_result < epsilon_err):
            break;

        print(f"Iteration -- {count}, error_result = {error_result}")
        count += 1

    return cluster_result

def KernelKmeans(img_data, gs, gc, init_method, cluster_num, epsilon_err):
    cluster_img_list = []
    (row, col, color_dim) = img_data.shape

    #Calculate the kernel function
    print(f">> CalculateKernelFunctions...")
    kernel = CalculateKernelFunctions(img_data, gs, gc, row, col, color_dim)

    #Calculate the kernel kmeans clustering
    print(f">> CenterInitialization...")
    (cluster_result, init_centroids) = CenterInitialization(init_method, cluster_num, row, col)

    #Perform the Kmeans algorithm
    print(f">> KmeansAlg...")
    cluster_result = KmeansAlg(cluster_result, init_centroids, img_data, row, col, cluster_num, kernel, epsilon_err)

    return (cluster_img_list)

def ReadInputFile(input_file):
    im      = Image.open(input_file)
    im_data = np.array(im)
    return im_data

#---------------Execution---------------#
if __name__ == '__main__':
    main()
