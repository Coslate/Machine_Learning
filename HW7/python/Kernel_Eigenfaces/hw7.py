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
    (input_training_dir, input_testing_dir, largest_k_pca, largest_k_lda, output_dir, k_nearest_neighbor, row, col, auto_constraint, pol_gamma, pol_coef0, pol_degree, rbf_gamma, is_debug) = ArgumentParser()

    print(f"> ReadInputFile...")
    train_img_data, train_img_label, row, col = ReadInputFile(input_training_dir, row, col)
    test_img_data, test_img_label, row, col   = ReadInputFile(input_testing_dir, row, col)

    print(f"> PerformPCA...")
    PerformPCA(train_img_data, train_img_label, test_img_data, test_img_label, largest_k_pca, k_nearest_neighbor, row, col, output_dir)

    print(f"> PerformKernelPCA...")
    print(f">> Linear Kernel...")
    PerformKernelPCA(train_img_data, train_img_label, test_img_data, test_img_label, largest_k_pca, k_nearest_neighbor, row, col, output_dir, 0, pol_gamma, pol_coef0, pol_degree, rbf_gamma)
    print(f">> Polynomial Kernel...")
    PerformKernelPCA(train_img_data, train_img_label, test_img_data, test_img_label, largest_k_pca, k_nearest_neighbor, row, col, output_dir, 1, pol_gamma, pol_coef0, pol_degree, rbf_gamma)
    print(f">> RBF Kernel...")
    PerformKernelPCA(train_img_data, train_img_label, test_img_data, test_img_label, largest_k_pca, k_nearest_neighbor, row, col, output_dir, 2, pol_gamma, pol_coef0, pol_degree, rbf_gamma)

    print(f"> PerformLDA...")
    PerformLDA(train_img_data, train_img_label, test_img_data, test_img_label, largest_k_lda, k_nearest_neighbor, row, col, output_dir, auto_constraint)

    if(is_debug):
        pass
        #print(f"len(train_img_data) = {len(train_img_data)}")
        #print(f"train_img_data[0].shape = {train_img_data[0].shape}")
        #print(f"len(test_img_data) = {len(test_img_data)}")
        #print(f"test_img_data[0].shape = {test_img_data[0].shape}")
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
    largest_k_pca       = 25
    largest_k_lda       = 14
    k_nearest_neighbor  = 5
    output_dir          = "./output"
    row                 = 29
    col                 = 41
    auto_constraint     = 1
    pol_gamma           = 0.01
    pol_coef0           = 0
    pol_degree          = 3
    rbf_gamma           = 0.000001
    is_debug            = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_training_dir",    "-itrd",     help="The directory name includes all training images.")
    parser.add_argument("--input_testing_dir",     "-ited",     help="The directory name includes all testing images.")
    parser.add_argument("--largest_k_pca",         "-lmk_pca",  help="The number of largest k eigenvectors that will be chosen in PCA. Default is 25.")
    parser.add_argument("--largest_k_lda",         "-lmk_lda",  help="The number of largest k eigenvectors that will be chosen in LDA. Default is 14.")
    parser.add_argument("--k_nearest_neighbor",    "-knn",      help="The number of nearest k neighbors for classifying wihich labels a testing object belongs to. Default is 10.")
    parser.add_argument("--row",                   "-row",      help="The resized row of the original image. Default is 29.")
    parser.add_argument("--col",                   "-col",      help="The resized col of the original image. Default is 41.")
    parser.add_argument("--auto_constraint",       "-auc",      help="Set 1 to auto constraints the maximal number of eigenvectors to be the cluster number. Set 0 for no such constraint. Default is 1.")
    parser.add_argument("--pol_gamma",             "-pg",       help="The gamma parameter of Polynomial Kernel. Default is 0.01.")
    parser.add_argument("--pol_coef0",             "-pc",       help="The coef0 parameter of Polynomial Kernel. Default is 0.")
    parser.add_argument("--pol_degree",            "-pd",       help="The degree parameter of Polynomial Kernel. Default is 3.")
    parser.add_argument("--rbf_gamma",             "-rg",       help="The gamma parameter of RBF Kernel. Default is 0.000001.")
    parser.add_argument("--output_dir",            "-odr",      help="The output directory of the result. Default is './output'")
    parser.add_argument("--is_debug",              "-isd",      help="1 for debug mode; 0 for normal mode.")

    args = parser.parse_args()

    if(args.input_training_dir):
        input_training_dir = args.input_training_dir
    if(args.input_testing_dir):
        input_testing_dir  = args.input_testing_dir
    if(args.largest_k_pca):
        largest_k_pca          = int(args.largest_k_pca)
    if(args.largest_k_lda):
        largest_k_lda          = int(args.largest_k_lda)
    if(args.k_nearest_neighbor):
        k_nearest_neighbor = int(args.k_nearest_neighbor)
    if(args.row):
        row                = int(args.row)
    if(args.col):
        col                = int(args.col)
    if(args.auto_constraint):
        auto_constraint    = int(args.auto_constraint)
    if(args.pol_gamma):
        pol_gamma          = float(args.pol_gamma)
    if(args.pol_coef0):
        pol_coef0          = float(args.pol_coef0)
    if(args.pol_degree):
        pol_degree         = float(args.pol_degree)
    if(args.rbf_gamma):
        rbf_gamma          = float(args.rbf_gamma)
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
        print(f"largest_k_pca        = {largest_k_pca}")
        print(f"largest_k_lda        = {largest_k_lda}")
        print(f"k_nearest_neighbor   = {k_nearest_neighbor}")
        print(f"row                  = {row}")
        print(f"col                  = {col}")
        print(f"auto_constraint      = {auto_constraint}")
        print(f"pol_gamma            = {pol_gamma}")
        print(f"pol_coef0            = {pol_coef0}")
        print(f"pol_degree           = {pol_degree}")
        print(f"rbf_gamma            = {rbf_gamma}")
        print(f"output_dir           = {output_dir}")
        print(f"is_debug             = {is_debug}")

    return (input_training_dir, input_testing_dir, largest_k_pca, largest_k_lda, output_dir, k_nearest_neighbor, row, col, auto_constraint, pol_gamma, pol_coef0, pol_degree, rbf_gamma, is_debug)

def FormMatrixLDA(data_matrix, label_matrix):
    N          = data_matrix.shape[0] #135
    orig_dimen = data_matrix.shape[1] #row*col

    #Get the number of each class in original high dimensions
    class_num = {}
    for i in range(N):
        if(class_num.get(label_matrix[i]) == None):
            class_num[label_matrix[i]]  = 1
        else:
            class_num[label_matrix[i]] += 1

    #Get the mean of each class
    class_mean = {}
    for i in range(N):
        if(not (label_matrix[i] in class_mean.keys())):
            class_mean[label_matrix[i]]  = data_matrix[i].astype(np.float64)
        else:
            class_mean[label_matrix[i]] += data_matrix[i].astype(np.float64)

    for i in class_mean.keys():
        class_mean[i] = class_mean[i]/class_num[i]

    #Calculate the within-class scatter
    sw = np.zeros((orig_dimen, orig_dimen), dtype=np.float64)
    for j in class_mean.keys(): #foreach class label j
        #Form Sj
        sj = np.zeros((orig_dimen, orig_dimen), dtype=np.float64)
        for i in range(N):
            if(label_matrix[i] == j):
                xi_min_u = data_matrix[i] - class_mean[j]
                sj      += xi_min_u.reshape(1, -1).T @ xi_min_u.reshape(1, -1)
        sw += sj

    #Calculate the total mean of x
    data_mean  = np.zeros(orig_dimen, dtype=np.float64) #1x(row*col)
    for i in range(N):
        for j in range(orig_dimen):
            data_mean[j] += data_matrix[i][j]
    data_mean = data_mean/N

    #Calculate the between-class scatter
    sb = np.zeros((orig_dimen, orig_dimen), dtype=np.float64)
    for j in class_mean.keys(): #foreach class label j
        #Form SBj
        mj_min_u = class_mean[j] - data_mean
        sbj      = class_num[j]*(mj_min_u.reshape(1, -1).T @ mj_min_u.reshape(1, -1))
        sb      += sbj

    #Calculate the (sw^-1)*sb
    w_mat = np.linalg.pinv(sw) @ sb

    return w_mat, len(class_num)

def FormKernelMatrixPCA(data_matrix, kernel_choice, pol_gamma, pol_coef0, pol_degree, rbf_gamma):
    N           = data_matrix.shape[1]
    data_matrix = data_matrix.astype(np.float64)

    #Calculate kernel matrix
    if(kernel_choice == 0):  #Linear
        #kernel_mat = data_matrix.T @ data_matrix
        kernel_mat  = data_matrix.T.dot(data_matrix)
    elif(kernel_choice == 1):#Polynomial
        kernel_mat = np.power((pol_gamma*(data_matrix.T @ data_matrix) + pol_coef0), pol_degree)
    elif(kernel_choice == 2):#RBF
        kernel_mat = np.exp(-rbf_gamma*(distance.cdist(data_matrix.T, data_matrix.T, 'sqeuclidean')))

    #Adjust center point
    one_n_mat    = np.full((N, N), 1/N, dtype=np.float64)
    kernel_c_mat = kernel_mat - one_n_mat@kernel_mat - kernel_mat@one_n_mat + one_n_mat@kernel_mat@one_n_mat

    return kernel_c_mat

def FormCovMatrixPCA(data_matrix):
    N          = data_matrix.shape[0] #135
    orig_dimen = data_matrix.shape[1] #row*col
    data_mean  = np.zeros(orig_dimen, dtype=np.float64) #1x(row*col)
    x_min_xmean= np.zeros((data_matrix.shape[1], data_matrix.shape[0]), dtype=np.float64) #x-ux

    #Calculate the mean
    for i in range(data_matrix.shape[0]):
        for j in range(data_matrix.shape[1]):
            data_mean[j] += data_matrix[i][j]
    data_mean = data_mean/N

    #Calculate x-ux
    for i in range(data_matrix.shape[0]):
        for j in range(data_matrix.shape[1]):
            x_min_xmean[j][i] = float(data_matrix[i][j]) - data_mean[j]

    cov_matrix = x_min_xmean @ x_min_xmean.T
    cov_matrix = cov_matrix/N

    return cov_matrix

def FormMaxKEigenMatrix(in_matrix, total_n, largest_k):
    eigen_values, eigen_vectors = np.linalg.eig(in_matrix)
#    eigen_matrix = np.zeros((total_n, largest_k), dtype=np.float64)
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

def ShowEigenFaces(eigen_mat, row, col, directory, out_file_name, largest_k):
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
        eigen_face = ((eigen_face - eigen_face.min()) * (1/(eigen_face.max() - eigen_face.min()) * 255))
        output_file_name = directory + "/"+out_file_name+"_w_largest_"+str(largest_k)+"_ev_eigen"+str(index+1)+".png"
        img_data_gen = eigen_face.astype(np.uint8)
        img_data_gen = Image.fromarray(img_data_gen)
        img_data_gen.save(output_file_name)

def FacesReconstruction(train_img_data, eigen_mat, directory, row, col, out_file_name, largest_k):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        #clean all gif and png files
        for zippath in glob.iglob(os.path.join(directory, '*origin*.png')):
            os.remove(zippath)

        for zippath in glob.iglob(os.path.join(directory, '*reconstruct*.png')):
            os.remove(zippath)

    orig_dim   = train_img_data.shape[1]
    new_dim    = eigen_mat.shape[1]
    picked_num = np.random.choice(train_img_data.shape[0], 10, replace=False)
    for i in range(10):
        picked_x      = train_img_data[picked_num[i]].astype(np.float64)
        reconstruct_z = picked_x.reshape(1, orig_dim) @ eigen_mat @ eigen_mat.T
        reconstruct_z = reconstruct_z.reshape(row, col)
        reconstruct_z = (reconstruct_z - reconstruct_z.min()) * (1/(reconstruct_z.max() - reconstruct_z.min()) * 255)

        out_file_origin_name = directory + "/"+out_file_name+"_w_largest_"+str(largest_k)+"_ev_origin"+str(i+1)+".png"
        img_data_gen = picked_x.reshape(row, col).astype(np.uint8)
        img_data_gen = Image.fromarray(img_data_gen)
        img_data_gen.save(out_file_origin_name)
        out_file_recon_name = directory + "/"+out_file_name+"_w_largest_"+str(largest_k)+"_ev_reconstruct"+str(i+1)+".png"
        img_data_gen = reconstruct_z.astype(np.uint8)
        img_data_gen = Image.fromarray(img_data_gen)
        img_data_gen.save(out_file_recon_name)

def ConvertToLowDimension(input_img_arr, eigen_mat):
    N          = input_img_arr.shape[0]
    orig_dim   = input_img_arr.shape[1]
    new_dim    = eigen_mat.shape[1]
    low_dim_z  = np.zeros((N, new_dim), dtype=np.float64)

    for i in range(N):
        picked_x      = input_img_arr[i].astype(np.float64)
        low_dim_z[i]  = picked_x.reshape(1, orig_dim) @ eigen_mat

    return low_dim_z

def Classification(test_img_data, test_img_label, train_img_data, train_img_label, eigen_mat, output_dir, row, col, k_nearest_neighbor, out_file_name, largest_k):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        #clean all gif and png files
        for zippath in glob.iglob(os.path.join(output_dir, '*.txt')):
            os.remove(zippath)

    #Transform the high dimension data to low dimension data
    train_z = ConvertToLowDimension(train_img_data, eigen_mat)
    test_z  = ConvertToLowDimension(test_img_data, eigen_mat)

    #Find the most frequent label among k-nearest neighbors
    correct = 0
    for test_idx in range(test_img_data.shape[0]):
        dist = np.zeros(train_img_data.shape[0], dtype=np.float64)
        for train_idx in range(train_img_data.shape[0]):
            dist[train_idx] = np.linalg.norm(test_z[test_idx]-train_z[train_idx])

        #Form k-nearest neighbors
        #print(f"dist.shape = {dist.shape}")
        #print(f"type(dist) = {type(dist)}")
        #print(f"dist = {dist}")

        k = 0
        k_min_index = np.zeros(k_nearest_neighbor, dtype=int)
        while(k < k_nearest_neighbor):
            k_min_index[k] = np.argmin(dist) #record the min k_nearest_neighbor index
            dist[k_min_index[k]] = np.inf
            k+=1

        #print(f"k_min_index.shape = {k_min_index.shape}")
        #print(f"type(k_min_index) = {type(k_min_index)}")
        #print(f"k_min_index = {k_min_index}")

        #Find the most frequent label among the k-nearest neighbors
        statistic_label = np.zeros(max(train_img_label)+1, dtype=int)
        for i in range(k_nearest_neighbor):
            #print(f"train_img_label[{k_min_index[i]}] = {train_img_label[k_min_index[i]]}")
            statistic_label[train_img_label[k_min_index[i]]] += 1
        most_likely_label = np.argmax(statistic_label)
        #print(f"train_img_label = {train_img_label}")
        #print(f"most_likely_label = {most_likely_label}")
        if(most_likely_label == test_img_label[test_idx]):
            correct += 1

    print(f"correct = {correct}")
    output_file_name = output_dir+"/"+out_file_name+"_classification.txt"
    lines = [f"Use first {largest_k} largest eigenvectors, ", f"knn with k = {k_nearest_neighbor}, ", f"#correctly classified = {correct}, ", f"#total = {test_img_label.shape[0]}, ", f"accuracy = {(correct/test_img_label.shape[0])*100}%"]
    with open(output_file_name, 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')

def PerformLDA(train_img_data, train_img_label, test_img_data, test_img_label, largest_k, k_nearest_neighbor, row, col, output_dir, auto_constraint):
    output_dir += "/LDA"

    #Form the covariance matrix
    w_mat, total_class_num  = FormMatrixLDA(train_img_data, train_img_label)

    #Constraint by rank of sb
    if(auto_constraint and (largest_k > total_class_num)):
        largest_k = total_class_num

    #Do the eigen decomposition on the covariance matrix
    eigen_mat  = FormMaxKEigenMatrix(w_mat, w_mat.shape[0], largest_k)

    #Show the eigen faces, the first largest eigen vectors
    ShowEigenFaces(eigen_mat, row, col, output_dir, "lda", largest_k)

    #Randomly pick 10 faces in train_img_data and do the reconstruction based on the eigen faces
    FacesReconstruction(train_img_data, eigen_mat, output_dir, row, col, "lda", largest_k)

    #Classification on testing images
    Classification(test_img_data, test_img_label, train_img_data, train_img_label, eigen_mat, output_dir, row, col, k_nearest_neighbor, "lda_knn", largest_k)

def PerformKernelPCA(train_img_data, train_img_label, test_img_data, test_img_label, largest_k, k_nearest_neighbor, row, col, output_dir, kernel_choice, pol_gamma, pol_coef0, pol_degree, rbf_gamma):
    if(kernel_choice == 0):
        kernel_name = "Linear"
    elif(kernel_choice == 1):
        kernel_name = "Polynomial"
    elif(kernel_choice == 2):
        kernel_name = "RBF"

    output_dir += "/Kernel_"+kernel_name+"_PCA"

    #Form the covariance matrix
    kernel_c_mat    = FormKernelMatrixPCA(train_img_data, kernel_choice, pol_gamma, pol_coef0, pol_degree, rbf_gamma)
    #Do the eigen decomposition on the covariance matrix
    eigen_mat  = FormMaxKEigenMatrix(kernel_c_mat, kernel_c_mat.shape[0], largest_k)

    #Show the eigen faces, the first largest eigen vectors
    ShowEigenFaces(eigen_mat, row, col, output_dir, "kernel_pca", largest_k)

    #Randomly pick 10 faces in train_img_data and do the reconstruction based on the eigen faces
    FacesReconstruction(train_img_data, eigen_mat, output_dir, row, col, "kernel_pca", largest_k)

    #Classification on testing images
    Classification(test_img_data, test_img_label, train_img_data, train_img_label, eigen_mat, output_dir, row, col, k_nearest_neighbor, "pca_knn", largest_k)

def PerformPCA(train_img_data, train_img_label, test_img_data, test_img_label, largest_k, k_nearest_neighbor, row, col, output_dir):
    output_dir += "/PCA"

    #Form the covariance matrix
    cov_mat    = FormCovMatrixPCA(train_img_data)

    #Do the eigen decomposition on the covariance matrix
    eigen_mat  = FormMaxKEigenMatrix(cov_mat, cov_mat.shape[0], largest_k)

    #Show the eigen faces, the first largest eigen vectors
    ShowEigenFaces(eigen_mat, row, col, output_dir, "pca", largest_k)

    #Randomly pick 10 faces in train_img_data and do the reconstruction based on the eigen faces
    FacesReconstruction(train_img_data, eigen_mat, output_dir, row, col, "pca", largest_k)

    #Classification on testing images
    Classification(test_img_data, test_img_label, train_img_data, train_img_label, eigen_mat, output_dir, row, col, k_nearest_neighbor, "pca_knn", largest_k)

def ReadInputFile(input_directory, row, col):
    #Get the list of image files under the directory
    img_filename = []
#    row = 100
#    col = 100
#    row = 29
#    col = 41
#    row = 29
#    col = 24

    for file in glob.glob(input_directory+"/*.pgm"):
        img_filename.append(file)

    img_data_array  = np.zeros((len(img_filename), row*col), dtype=np.uint8)
    img_label_array = np.zeros(len(img_filename), dtype=int)
    update_done     = 0

    for i, file_name in enumerate(img_filename):
        im                 = Image.open(file_name)
        org_col, org_row   = im.size

        if(update_done == 0):
            if(row > org_row):
                row = org_row
            if(col > org_col):
                col = org_col
            update_done = 1

        new_im             = im.resize((col, row))
        im_data            = np.array(new_im)
        img_data_array[i]  = im_data.reshape(1, -1)
        match_result       = re.match(r'.*\/subject(\d+)\..*', file_name)
        img_label_array[i] = match_result.group(1)

    return img_data_array, img_label_array, row, col

#---------------Execution---------------#
if __name__ == '__main__':
    main()
