#! /usr/bin/env python3
'''
    Author      : BCC
    Date        : 2022/05/02
'''

import argparse
import math
import sys
import re
import os
import glob
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
    (input_img1, input_img2, gs, gc, init_method, cluster_num, epsilon_err, directory, gen_first_k_iter, is_debug) = ArgumentParser()

    print(f"> ReadInputFile...")
    img_data1 = ReadInputFile(input_img1)
    img_data2 = ReadInputFile(input_img2)

    print(f"> KernelKmeans...")
    (cluster_img1_list) = KernelKmeans(img_data1, gs, gc, init_method, cluster_num, epsilon_err)
    (cluster_img2_list) = KernelKmeans(img_data2, gs, gc, init_method, cluster_num, epsilon_err)

    print(f"> OutputResult...")
    OutputResult(cluster_img1_list, cluster_img2_list, directory, init_method, cluster_num, gen_first_k_iter)

    if(is_debug):
        print(f"im_data1.type = {img_data1.shape}")
        print(f"im_data2.type = {img_data2.shape}")
        img1 = Image.fromarray(img_data1)
        img2 = Image.fromarray(img_data2)
        img1.save(f'{directory}/test_img1.png')
        img2.save(f'{directory}/test_img2.png')
        #fig, ax = plt.subplots(1,2)
        #ax[0].imshow(img1)
        #ax[1].imshow(img2)
        #plt.show()
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
    directory           = "./output"
    gen_first_k_iter    = 2
    is_debug            = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_img1",   "-img1",     help="The file name of the first input image.")
    parser.add_argument("--input_img2",   "-img2",     help="The file name of the second input image.")
    parser.add_argument("--gs",           "-gs",       help="The gamma hyper parameter for RBF kernel of space.")
    parser.add_argument("--gc",           "-gc",       help="The gamma hyper parameter for RBF kernel of color.")
    parser.add_argument("--init_method",  "-init_m",   help="The method for choosing initial centroids for clustering. Set 0 for random selection. Set 1 for kmeans++ selection. Default 0.")
    parser.add_argument("--cluster_num",  "-cn",       help="The number of clusters you want. Default is 2.")
    parser.add_argument("--epsilon_err",  "-eps",      help="The error in neighbored step that Kmeans can terminate. Default is 0.0001.")
    parser.add_argument("--directory",        "-dir",       help="The output directory of the result. Default is './output'")
    parser.add_argument("--gen_first_k_iter", "-gfk",       help="The number of the iteraions of the first k stages of kmeans will be output. Default is 2.")
    parser.add_argument("--is_debug",          "-isd",      help="1 for debug mode; 0 for normal mode.")

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
    if(args.directory):
        directory         = args.directory
    if(args.gen_first_k_iter):
        gen_first_k_iter  = int(args.gen_first_k_iter)
    if(args.is_debug):
        is_debug          = int(args.is_debug)

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
        print(f"directory    = {directory}")
        print(f"gen_first_k_iter = {gen_first_k_iter}")
        print(f"is_debug     = {is_debug}")

    return (input_img1, input_img2, gs, gc, init_method, cluster_num, epsilon_err, directory, gen_first_k_iter, is_debug)

def OutputResult(cluster_img1_list, cluster_img2_list, directory, init_method, cluster_num, gen_first_k_iter):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        #clean all gif and png files
        for zippath in glob.iglob(os.path.join(directory, '*result*.png')):
            os.remove(zippath)

        for zippath in glob.iglob(os.path.join(directory, '*result*.gif')):
            os.remove(zippath)

    if(init_method == 0):
        mode = "random"
    elif(init_method == 1):
        mode = "kmeans++"

    if(gen_first_k_iter > (len(cluster_img1_list)-1)):
        gen_first_k_iter = len(cluster_img1_list)-1

    for i in range(gen_first_k_iter):
        #Output Iteration i result
        out_img1_file_name = directory + "/kernelkmeans_result_cluster_"+str(cluster_num)+"_mode_"+str(mode)+"_iteration_"+str(i)+"_img1.png"
        out_img2_file_name = directory + "/kernelkmeans_result_cluster_"+str(cluster_num)+"_mode_"+str(mode)+"_iteration_"+str(i)+"_img2.png"

        cluster_img1_list[i].save(out_img1_file_name)
        cluster_img2_list[i].save(out_img2_file_name)

    #Output Iteration n result
    out_img1_file_name = directory + "/kernelkmeans_result_cluster_"+str(cluster_num)+"_mode_"+str(mode)+"_iteration_"+str(len(cluster_img1_list)-1)+"_img1.png"
    out_img2_file_name = directory + "/kernelkmeans_result_cluster_"+str(cluster_num)+"_mode_"+str(mode)+"_iteration_"+str(len(cluster_img2_list)-1)+"_img2.png"

    cluster_img1_list[-1].save(out_img1_file_name)
    cluster_img2_list[-1].save(out_img2_file_name)

    #Output GIF files
    out_gif1_file_name = directory+"/kernelkmeans_result_cluster_"+str(cluster_num)+"_mode_"+mode+"_img1.gif"
    out_gif2_file_name = directory+"/kernelkmeans_result_cluster_"+str(cluster_num)+"_mode_"+mode+"_img2.gif"

    cluster_img1_list[0].save(out_gif1_file_name, save_all=True, append_images=cluster_img1_list[1:], optimize=False, duration=150, loop=0)
    cluster_img2_list[0].save(out_gif2_file_name, save_all=True, append_images=cluster_img2_list[1:], optimize=False, duration=150, loop=0)

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
        for has_chosen in range(cluster_num):
            new_init_center = np.random.choice(row*col, 1, replace=False)
            new_init_i = math.floor(new_init_center[0]/col)
            new_init_j = new_init_center[0]%col
            init_centroids[has_chosen][0] = new_init_j
            init_centroids[has_chosen][1] = new_init_i

    elif(init_method == 1):
        #Kmeans++
        #Step1 - Choose an initial center c1 uniformly.
        new_init_center = np.random.choice(row*col, 1, replace=False)
        new_init_i = math.floor(new_init_center[0]/col)
        new_init_j = new_init_center[0]%col
        init_centroids[0][0] = new_init_j
        init_centroids[0][1] = new_init_i
        has_chosen = 1

        #Step2 - Choose next center ci with porbability = d(x)^2/(sum(d(x)^2).
        while(has_chosen < cluster_num):
            #Find the shortest distance of every pixel to all the chosen center
            dist_to_center_shortest = np.zeros((row, col, 1), dtype=np.float64)
            for i in range(row):
                for j in range(col):
                    coords = np.array([i, j])
                    min_dist = np.inf
                    for chosen_center_idx in range(has_chosen):
                        chosen_center = init_centroids[chosen_center_idx]
                        dist = math.pow(np.linalg.norm((chosen_center - coords)), 2)
                        if(dist < min_dist):
                            min_dist = dist
                    dist_to_center_shortest[i][j] = min_dist

            #Construct the probability : d(x)^2/sum(d(x)^2)
            dist_to_center_shortest = dist_to_center_shortest.reshape(row*col)
            dist_to_center_shortest /= np.sum(dist_to_center_shortest)
            new_init_center = np.random.choice(row*col, 1, replace=False, p=dist_to_center_shortest)
            new_init_i = math.floor(new_init_center[0]/col)
            new_init_j = new_init_center[0]%col

            if [new_init_j, new_init_i] in init_centroids.tolist():
                continue
            else:
                init_centroids[has_chosen][0] = new_init_j
                init_centroids[has_chosen][1] = new_init_i
                has_chosen += 1

    #Initialize to the center points
    for clus_num, center in enumerate(init_centroids):
        i = center[1]
        j = center[0]
        pos = i*col + j
        cluster_result[int(pos)] = clus_num

    return cluster_result, init_centroids

    '''
def CalculateEachClusterGroup(cluster_result, cluster_num):
    cluster_group = [[] for x in range(cluster_num)]
    cluster_size  = np.zeros(cluster_num, dtype=np.float64)

    for pixel_pos, assigned_cluster in enumerate(cluster_result):
        if(assigned_cluster == -1):
            continue

        cluster_size[int(assigned_cluster)] += 1
        cluster_group[int(assigned_cluster)].append(pixel_pos)

    return cluster_size, cluster_group
    '''

@nb.jit(nopython=True, nogil=True)
def CalculateEachClusterGroup(cluster_result, cluster_num):
    cluster_size  = np.zeros(cluster_num, dtype=np.float64)

    for assigned_cluster in cluster_result:
        if(assigned_cluster == -1):
            continue

        cluster_size[int(assigned_cluster)] += 1

    #Prevent dividing by zero
    cluster_size[cluster_size == 0] = 1
    return cluster_size

@nb.jit(nopython=True, nogil=True)
def SumWithOtherClusterMem(pixel_pos, cluster_result, kernel, total_size_image, clus_label):
    kernel_sum = 0
    for pixel_other_pos in range(total_size_image):
        if(cluster_result[pixel_other_pos] == clus_label):
            kernel_sum += kernel[pixel_pos][pixel_other_pos]

    return kernel_sum

@nb.jit(nopython=True, nogil=True)
def SumTwoClusterMem(cluster_result, cluster_num, kernel, total_size_image):
    kernel_sum   = np.zeros(cluster_num)

    for i in range(total_size_image):
        for j in range(total_size_image):
            if(cluster_result[i] == cluster_result[j]):
                kernel_sum[int(cluster_result[i])] += kernel[i][j]


    return kernel_sum

@nb.jit(nopython=True, nogil=True)
def UpdateClusterIndicator(new_cluster_result, cluster_result, total_size_image, kernel, init_centroids, cluster_size, cluster_num):
    sum_two_mem        = SumTwoClusterMem(cluster_result, cluster_num, kernel, total_size_image)

    #Assign each pixel to its closest center
    for pixel_pos in range(total_size_image):
        distance = np.zeros(cluster_num, dtype=np.float64)

        #foreach of the cluster
        for clus_label in range(cluster_num):
            sum_with_other_mem = SumWithOtherClusterMem(pixel_pos, cluster_result, kernel, total_size_image, clus_label)

            distance[clus_label] = kernel[pixel_pos][pixel_pos] - (2.0/cluster_size[clus_label])*sum_with_other_mem + (1/(cluster_size[clus_label]*cluster_size[clus_label]))*sum_two_mem[clus_label]

        new_cluster_result[pixel_pos] = np.argmin(distance)

    return new_cluster_result


def KmeansClustering(cluster_result, init_centroids, cluster_num, kernel, row, col):
    cluster_size  = CalculateEachClusterGroup(cluster_result, cluster_num)

    #Assign each pixel to its closest center
    total_size_image   = row*col
    new_cluster_result = np.zeros(total_size_image)
    new_cluster_result = UpdateClusterIndicator(new_cluster_result, cluster_result, total_size_image, kernel, init_centroids, cluster_size, cluster_num)

    return new_cluster_result


def GenerateImg(cluster_result, row, col, color_arr):
    img_data_gen = np.zeros((row*col, 3))
    for i in range(row):
        for j in range(col):
            pos = i*col + j
            if(cluster_result[pos] == -1):
                img_data_gen[pos] = np.array([0, 0, 0])
            else:
                img_data_gen[pos] = color_arr[int(cluster_result[pos])]

    img_data_gen = img_data_gen.reshape(row, col, 3)
    img_data_gen = img_data_gen.astype(np.uint8)
    return Image.fromarray(img_data_gen)

def GenerateColor(cluster_num):
    color_arr = np.zeros((cluster_num, 3))

    if(cluster_num == 1):
        color_arr[0] = np.array([255, 0, 0]) #R
    elif(cluster_num == 2):
        color_arr[0] = np.array([255, 0, 0]) #R
        color_arr[1] = np.array([0, 255, 0]) #G
    elif(cluster_num == 3):
        color_arr[0] = np.array([255, 0, 0]) #R
        color_arr[1] = np.array([0, 255, 0]) #G
        color_arr[2] = np.array([0, 0, 255]) #B
    elif(cluster_num > 3):
        color_arr[0] = np.array([255, 0, 0]) #R
        color_arr[1] = np.array([0, 255, 0]) #G
        color_arr[2] = np.array([0, 0, 255]) #B

        for left_color in range(cluster_num-3):
            color_pos = left_color+3
            while(True):
                r = np.random.randint(0, 255+1)
                g = np.random.randint(0, 255+1)
                b = np.random.randint(0, 255+1)

                color_comb = np.array([r, g, b])
                equal      = False
                for x in range(3+left_color):
                    if((color_comb == color_arr).all()):
                        equal = True
                        break;

                if(not equal):
                    color_arr[color_pos] = color_comb
                    break;
    return color_arr

def KmeansAlg(cluster_result, init_centroids, img_data, row, col, cluster_num, kernel, epsilon_err):
    cluster_img_list = []
    count            = 1
    color_arr        = GenerateColor(cluster_num)

    clustered_img    = GenerateImg(cluster_result, row, col, color_arr)
    cluster_img_list.append(clustered_img)
    print(f">>> KmeansClustering...")
    while(True):
        #Classify all samples according to the closest mean(init_centroids)
        new_cluster_result = KmeansClustering(cluster_result, init_centroids, cluster_num, kernel, row, col)
        error_result       = np.linalg.norm((new_cluster_result - cluster_result))

        cluster_result   = new_cluster_result.copy()
        clustered_img    = GenerateImg(cluster_result, row, col, color_arr)
        cluster_img_list.append(clustered_img)

        if(error_result < epsilon_err):
            print(f"-------------------Achieved!!!---------------------")
            print(f"Iteration -- {count}, error_result = {error_result}")
            print(f"---------------------------------------------------")
            break;

        print(f"Iteration -- {count}, error_result = {error_result}")
        count += 1

    return cluster_img_list, cluster_result

def KernelKmeans(img_data, gs, gc, init_method, cluster_num, epsilon_err):
    (row, col, color_dim) = img_data.shape

    #Calculate the kernel function
    print(f">> CalculateKernelFunctions...")
    kernel = CalculateKernelFunctions(img_data, gs, gc, row, col, color_dim)

    #Calculate the kernel kmeans clustering
    print(f">> CenterInitialization...")
    (cluster_result, init_centroids) = CenterInitialization(init_method, cluster_num, row, col)

    #Perform the Kmeans algorithm
    print(f">> KmeansAlg...")
    (cluster_img_list, cluster_result) = KmeansAlg(cluster_result, init_centroids, img_data, row, col, cluster_num, kernel, epsilon_err)

    return cluster_img_list

def ReadInputFile(input_file):
    im      = Image.open(input_file)
    im_data = np.array(im)
    return im_data

#---------------Execution---------------#
if __name__ == '__main__':
    main()
