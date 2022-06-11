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
import pylab
import io
from scipy.spatial import distance
from PIL import Image
from matplotlib.pyplot import MultipleLocator #for setting of scale of separating along with x-axis & y-axis.
from matplotlib.colors import ListedColormap

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
    (input_img_file, input_label_file, mode, perplexity, directory, early_termin, early_termin_epsilon, is_debug) = ArgumentParser()

    print(f"> Load Training Data...")
    X = np.loadtxt(input_img_file)
    labels = np.loadtxt(input_label_file)

    print(f"> Perform t-SNE/s-SNE...")
    Y, proc_img, proc_img_fixed, P, Q_fin, C_optimized = PerformSNE(X, labels, 2, 50, mode, perplexity, early_termin, early_termin_epsilon)

    print(f"> Show Result...")
    OutputGIF(proc_img, proc_img_fixed, directory, mode, perplexity)
    OutputError(C_optimized, directory, mode, perplexity)
    DisplaySimilarity(P,     labels, mode, perplexity, directory, 1)
    DisplaySimilarity(Q_fin, labels, mode, perplexity, directory, 0)

    if(is_debug):
        pass
        #print(f"im_data1.type = {img_data1.shape}")
        #print(f"im_data2.type = {img_data2.shape}")
        #img1 = Image.fromarray(img_data1)
        #img2 = Image.fromarray(img_data2)
        #img1.save(f'{directory}/test_img1.png')
        #img2.save(f'{directory}/test_img2.png')
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
    input_img_file      = None
    input_label_file    = None
    mode                = 1
    perplexity          = 20.0
    directory           = "./output"
    early_termin        = 0
    early_termin_epsilon= 0.000001
    is_debug            = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_img_file",   "-img",     help="The file name of the input image.")
    parser.add_argument("--input_label_file", "-ilb",     help="The file name of the input label.")
    parser.add_argument("--mode",             "-mode",    help="Set 1 to perform t-SNE algorithm. Set 0 to perform Symmetric-SNE algorithm. Default is 1")
    parser.add_argument("--perplexity",       "-perp",    help="The perplexity used in SNE algorithm. Default is 20.0")
    parser.add_argument("--directory",        "-dir",     help="The output directory of the result. Default is './output'")
    parser.add_argument("--early_termin",     "-et",      help="Set 0 to wait 1000 iteration fo t-SNE/Symmetric-SNE algoruthm. Set 1 to early terminate when error is smaller than 'eraly_termin_epsilon'. Default is 0.")
    parser.add_argument("--early_termin_epsilon", "-et_eps",  help="The early terminate error that can be tolerated. Default is 1e-6.")
    parser.add_argument("--is_debug",         "-isd",     help="1 for debug mode; 0 for normal mode.")

    args = parser.parse_args()

    if(args.input_img_file):
        input_img_file       = args.input_img_file
    if(args.input_label_file):
        input_label_file     = args.input_label_file
    if(args.mode):
        mode                 = int(args.mode)
    if(args.perplexity):
        perplexity           = float(args.perplexity)
    if(args.directory):
        directory            = args.directory
    if(args.early_termin):
        early_termin         = int(args.early_termin)
    if(args.early_termin_epsilon):
        early_termin_epsilon = float(args.early_termin_epsilon)
    if(args.is_debug):
        is_debug             = int(args.is_debug)

    if(input_img_file == None):
        print(f"Error: You should set '--input_img_file' or '-img' for the file name of the input image.")
        sys.exit()

    if(input_label_file == None):
        print(f"Error: You should set '--input_label_file' or '-ilb' for the file name of the input label.")
        sys.exit()

    if(is_debug):
        print(f"input_img_file        = {input_img_file}")
        print(f"input_label_file      = {input_label_file}")
        print(f"mode                  = {mode}")
        print(f"perplexity            = {perplexity}")
        print(f"directory             = {directory}")
        print(f"early_termin          = {early_termin}")
        print(f"early_termin_epsilon  = {early_termin_epsilon}")
        print(f"is_debug              = {is_debug}")

    return (input_img_file, input_label_file, mode, perplexity, directory, early_termin, early_termin_epsilon, is_debug)

def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def PerformSNE(X=np.array([]), labels=np.array([]), no_dims=2, initial_dims=50, mode=1, perplexity=30.0, early_termin=0, early_termin_epsilon=0.000001):
    """
        Runs t-SNE/s-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # For early terminating
    C_prev = 0.
    C_optimized = 0.

    # For displaying
    Q_fin = np.zeros((n, n), dtype=np.float64)

    # GIF of the procedure
    proc_img = []
    proc_img_fixed = []

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.									# early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        if(mode == 1):
            #t-SNE
            num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        else:
            #s-SNE
            num = np.exp(-1. * np.add(np.add(num, sum_Y).T, sum_Y))

        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        if(mode == 1):
            #t-SNE
            for i in range(n):
                dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)
        else:
            #s-SNE
            for i in range(n):
                dY[i, :] = np.sum(np.tile(PQ[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Recrd the procedure
        proc_img.append(TransformToImg(Y, labels, mode, perplexity, 0, None))
        if(mode == 1):
            proc_img_fixed.append(TransformToImg(Y, labels, mode, perplexity, 1, [-150, 150]))
        else:
            proc_img_fixed.append(TransformToImg(Y, labels, mode, perplexity, 1, [-10, 10]))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            C_optimized = C
            print("Iteration %d: error is %f" % (iter + 1, C))

            if(early_termin == 1):
                if(abs(C - C_prev) < early_termin_epsilon):
                    P = P / 4.
                    Q_fin = Q.copy()
                    break

                C_prev = C

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

        # For displaying
        Q_fin = Q.copy()

    # Return solution
    return Y, proc_img, proc_img_fixed, P, Q_fin, C_optimized

def TransformToImg(Y, labels, mode, perplexity, fixed_range, range_to_plot):
    plt.clf()

    if(fixed_range == 1):
        plt.xlim(range_to_plot)
        plt.ylim(range_to_plot)

    plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
    if(mode == 1):
        title_str = f"t-SNE with perplexity = {perplexity}"
    else:
        title_str = f"Symmetric-SNE with perplexity = {perplexity}"

    plt.title(f'{title_str}')
    plt.tight_layout()

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    im = Image.open(img_buf)
    return im

def OutputError(C_optimized, directory, mode, perplexity):
    if(mode == 1):
        title_str = f"t-SNE_perplexity_{perplexity}"
    else:
        title_str = f"Symmetric-SNE_perplexity_{perplexity}"

    file_name = title_str+"_final_error_"+str(C_optimized)+".txt"

    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        #clean all gif and png files
        for zippath in glob.iglob(os.path.join(directory, file_name)):
            os.remove(zippath)

    output_file_name = directory+"/"+file_name
    if(mode == 1):
        lines = [f"t-SNE, perplexity = {perplexity}, ", f"final error of objective function C = {C_optimized}"]
    else:
        lines = [f"Symmetric-SNE, perplexity = {perplexity}, ", f"final error of objective function C = {C_optimized}"]

    with open(output_file_name, 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')

def OutputGIF(proc_img, proc_img_fixed, directory, mode, perplexity):
    if(mode == 1):
        title_str = f"t-SNE_perplexity_{perplexity}"
        fixed_range = "-150_to_150"
    else:
        title_str = f"Symmetric-SNE_perplexity_{perplexity}"
        fixed_range = "-10_to_10"

    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        #clean all gif and png files
        for zippath in glob.iglob(os.path.join(directory, title_str+'.gif')):
            os.remove(zippath)
        for zippath in glob.iglob(os.path.join(directory, title_str+'_result.png')):
            os.remove(zippath)

    #Output GIF files & resulted image file
    out_gif_file_name = directory+"/"+title_str+".gif"
    proc_img[0].save(out_gif_file_name, save_all=True, append_images=proc_img[1:], optimize=False, duration=100, loop=0)
    proc_img[-1].save(f"{directory}/{title_str}_result.png")
    proc_img_fixed[-1].save(f"{directory}/{title_str}_fixed_range_{fixed_range}_result.png")

def DisplaySimilarity(prob_matrix, labels, mode, perplexity, directory, h_dim):
    if(mode == 1):
        if(h_dim == 1):
            title_str = f"t-SNE, perplexity = {perplexity}, High-D Similarity"
        else:
            title_str = f"t-SNE, perplexity = {perplexity}, Low-D Similarity"
    else:
        if(h_dim == 1):
            title_str = f"Symmetric-SNE, perplexity = {perplexity}, High-D Similarity"
        else:
            title_str = f"Symmetric-SNE, perplexity = {perplexity}, Low-D Similarity"

    #Re-arranging the P and Q matrix according to labels
    index = np.argsort(labels)
    scaled_p = np.log(prob_matrix)
    re_arranged_p = scaled_p[index][:, index]

    #Plot the similarity matrix according to Pij and Qij
    plt.clf()
    plt.figure(1)

    #img  = plt.imshow(re_arranged_p, cmap='RdYlBu_r', vmin=re_arranged_p.min(), vmax=re_arranged_p.max())
    img  = plt.imshow(re_arranged_p, cmap='RdYlBu_r')
    plt.colorbar(img)
    plt.title(f"{title_str}")

    #Output the image of the similarity matrix of Pij and Qij
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    im = Image.open(img_buf)

    if(mode==1):
        if(h_dim == 1):
            file_str = "t-SNE_perplexity_"+str(perplexity)+"_high_D_similarity.png"
        else:
            file_str = "t-SNE_perplexity_"+str(perplexity)+"_low_D_similarity.png"
        out_img_file_name = directory + "/"+file_str
    else:
        if(h_dim == 1):
            file_str = "Symmetric-SNE_perplexity_"+str(perplexity)+"_high_D_similarity.png"
        else:
            file_str = "Symmetric-SNE_perplexity_"+str(perplexity)+"_low_D_similarity.png"
        out_img_file_name = directory + "/"+file_str

    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        #clean all gif and png files
        for zippath in glob.iglob(os.path.join(directory, file_str+'.png')):
            os.remove(zippath)

    im.save(out_img_file_name)

#---------------Execution---------------#
if __name__ == '__main__':
    main()
