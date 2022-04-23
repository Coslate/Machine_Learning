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
import random
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import numba as nb
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
    print(f"> Parsing Arguments...")
    (infile_train_label, infile_train_image, use_color, EPSILON, termin_cnt, PSEUDO_CNST_LAMBDA, PSEUDO_CNST_P, pseudo_p_method, use_pseudo_w, use_pseudo_lambda, use_pseudo_p, is_debug) = ArgumentParser()

    #Get the input data point
    print(f"> Reading Training Label Files...")
    train_y = ReadLabelFile(infile_train_label, 2049, is_debug)
    print(f"> Reading Training Image Files...")
    train_x = ReadImageFile(infile_train_image, 2051, is_debug)
    train_y_array = np.array(train_y)
    train_x_array = np.array(train_x)
    rows = len(train_x[0])
    cols = len(train_x[0][0])
    num_of_images = len(train_x_array)

    #Transform input gray-level image to 2 bins image
    print(f"> Transforming Training Image to Binary Images...")
    binary_x = Transform2Bins(train_x_array, train_y_array, is_debug, num_of_images, rows, cols)

    print(f"> Performing EM Algorithms...")
    (lambda_set, p, total_iteration) = PerformEMAlgorithm(binary_x, rows, cols, EPSILON, termin_cnt, use_color, PSEUDO_CNST_LAMBDA, PSEUDO_CNST_P, pseudo_p_method, num_of_images, use_pseudo_w, use_pseudo_lambda, use_pseudo_p)

    #Get the mapping from the clustering to real class
    (mapping, count) = CalculateMapping(binary_x, train_y_array, lambda_set, p, num_of_images, rows, cols)

    if(is_debug):
        print(f"mapping = {mapping}")
        PrintMatrix(count, "count")

    #Display RealClass
    DisplayRealClassImagination(p, mapping, use_color, rows, cols)

    #Predict the result
    predict = PredictResult(binary_x, train_y_array, lambda_set, p, num_of_images, rows, cols, mapping)
    if(is_debug):
        PrintMatrix(predict, "predict")

    #Calculate Confusion Matrix
    CalConfusionMatrixPrintResult(predict, num_of_images, total_iteration)


    #Print the debug messages when necessary
    if(is_debug):
        '''
        print(f"all_disximage_storing statistics:")
        for target, image_list in all_disximage_storing.items():
            print(f"target : {target}, number of images = {len(image_list)}")
            print(f"bin information: ")
            bin_value_stat = [0 for i in range(32)]
            for image in image_list:
                for i in range(len(image)):
                    for j in range(len(image[0])):
                        bin_value_stat[image[i][j]] += 1

            for index, value in enumerate(bin_value_stat):
                print(f"total number of bin {index} = {value}")
            print(f"total number of pixels = {sum(bin_value_stat)}")
            print(f"-----------------------------")

        print(f"all_disximage_storing every 28x28 bin statistics")
        for target, image_list in all_disximage_storing.items():
            print(f"target : {target}, number of images = {len(image_list)}")
            print(f"bin information: ")
            bin_map = [[[0 for k in range(32)] for j in range(28)] for i in range(28)]
            for image in image_list:
                for i in range(len(image)):
                    for j in range(len(image[0])):
                        bin_map[i][j][image[i][j]] += 1

            for i in range(28):
                for j in range(28):
                    print(f"i = {i}, j = {j}, bin_map[{i}][{j}] = {bin_map[i][j]}")
            print(f"---------------------------------------")
        print(f"total training images number = {len(train_x)}")

        i = 0
        print(f"train_y[{i}] = {train_y[i]}")
        ShowImage(train_x[i])

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
        '''


#########################
#     Sub-Routine       #
#########################
def ArgumentParser():
    infile_train_label= None
    infile_train_image= None
    use_color         = None
    EPSILON           = 0.1
    termin_cnt        = 30
    PSEUDO_CNST_LAMBDA= 0.1
    PSEUDO_CNST_P     = 1/60000
    pseudo_p_method   = 0
    use_pseudo_w      = 0
    use_pseudo_lambda = 0
    use_pseudo_p      = 0
    is_debug          = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--infile_train_label", "-inf_tr_label", help="Should set the input training label file.")
    parser.add_argument("--infile_train_image", "-inf_tr_image", help="Should set the input training image file.")
    parser.add_argument("--use_color", "-uc", help="Set 1 to use color for display the imagination of Naive Baye's Classifier. Set 0 to print in plain text.")
    parser.add_argument("--epsilon", "-eps", help="Set the error precision of the neighbored steps for terminating the EM algorithm. Default 1e^-1.")
    parser.add_argument("--termin_cnt", "-tcn", help="Set the maximum iteration number for EM algorithm. Default 30.")
    parser.add_argument("--pseudo_cnst_lambda", "-pse_lamd", help="Set the pseudo const lambda for 0 lambda value. Default 0.1.")
    parser.add_argument("--pseudo_cnst_p", "-pse_p", help="Set the pseudo const p for 0 Bernouli(p) value. Default 1/784.")
    parser.add_argument("--pseudo_p_method", "-p_mth", help="Set 1 to automatically replace the 0 value of p with min non-zero min neighbor values of p[my_class]. Default 0.")
    parser.add_argument("--use_pseudo_w", "-u_pse_w", help="Set 1 to automatically replace the 0 value of w with 0.1. Default 0.")
    parser.add_argument("--use_pseudo_lambda", "-u_pse_l", help="Set 1 to use pseudo value of lambda if encountering 0 value. Default 0.")
    parser.add_argument("--use_pseudo_p", "-u_pse_p", help="Set 1 to use pseudo value of p if encountering 0 value. Default 0.")
    parser.add_argument("--is_debug", "-isd", help="1 for debug mode; 0 for normal mode.")

    args = parser.parse_args()

    if args.infile_train_label:
        infile_train_label = args.infile_train_label
    if args.infile_train_image:
        infile_train_image = args.infile_train_image
    if args.use_color:
        use_color  = int(args.use_color)
    if args.epsilon:
        EPSILON    = float(args.epsilon)
    if args.termin_cnt:
        termin_cnt = int(args.termin_cnt)
    if args.pseudo_cnst_lambda:
        PSEUDO_CNST_LAMBDA = float(args.pseudo_cnst_lambda)
    if args.pseudo_cnst_p:
        PSEUDO_CNST_P = float(args.pseudo_cnst_p)
    if args.pseudo_p_method:
        pseudo_p_method   = int(args.pseudo_p_method)
    if args.use_pseudo_w:
        use_pseudo_w   = int(args.use_pseudo_w)
    if args.use_pseudo_lambda:
        use_pseudo_lambda   = int(args.use_pseudo_lambda)
    if args.use_pseudo_p:
        use_pseudo_p   = int(args.use_pseudo_p)
    if args.is_debug:
        is_debug   = int(args.is_debug)

    if(infile_train_label ==  None):
        print(f"Error: You should set input file name with for training label '--infile_train_label' or '-inf_tr_label'")
        sys.exit()

    if(infile_train_label ==  None):
        print(f"Error: You should set input file name with for training image '--infile_train_label' or '-inf_tr_image'")
        sys.exit()

    if(use_color == None):
        use_color = 1

    if(is_debug):
        print(f"infile_train_label = {infile_train_label}")
        print(f"infile_train_image = {infile_train_image}")
        print(f"use_color          = {use_color}")
        print(f"EPSILON            = {EPSILON}")
        print(f"termin_cnt         = {termin_cnt}")
        print(f"PSEUDO_CNST_LAMBDA = {PSEUDO_CNST_LAMBDA}")
        print(f"PSEUDO_CNST_P      = {PSEUDO_CNST_P}")
        print(f"pseudo_p_method    = {pseudo_p_method}")
        print(f"use_pseudo_w       = {use_pseudo_w}")
        print(f"use_pseudo_lambda  = {use_pseudo_lambda}")
        print(f"use_pseudo_p       = {use_pseudo_p}")

    return (infile_train_label, infile_train_image, use_color, EPSILON, termin_cnt, PSEUDO_CNST_LAMBDA, PSEUDO_CNST_P, pseudo_p_method, use_pseudo_w, use_pseudo_lambda, use_pseudo_p, is_debug)

def DisplayOneClass(p_of_class, my_class, use_color, rows=28, cols=28):
    print(f"")

    print(f"class {my_class}: ")
    for i in range(rows):
        for j in range(cols):
            if(p_of_class[i][j] != 0):
                if(use_color==1):
                    print(color.CYAN+f"{p_of_class[i][j]:5.10f} "+color.END, end='')
                else:
                    print(f"{p_of_class[i][j]:5.10f} ", end='')
            else:
                print(f"0 ", end='')
        print(f"")
    print(f"")

def DisplayRealClassImagination(p, mapping, use_color, rows=28, cols=28):
    print(f"")
    print(f"--------------------------------------------")
    print(f"--------------------------------------------")
    print(f"")

    for my_class in range(10):
        #Find the class cluster result in my_class
        cluster_p = 0
        for cluster in range(10):
            if(int(mapping[cluster]) == my_class):
                cluster_p = cluster

        print(f"labeled class {my_class}: ")
        for i in range(rows):
            for j in range(cols):
                if(p[cluster_p][i][j] >= 0.5):
                    if(use_color==1):
                        print(color.CYAN+"1 "+color.END, end='')
                    else:
                        print("1 ", end='')
                else:
                    print(f"0 ", end='')
            print(f"")
        print(f"")

def DisplayImagination(p, use_color, rows=28, cols=28):
    print(f"")

    for my_class in range(10):
        print(f"class {my_class}: ")
        for i in range(rows):
            for j in range(cols):
                if(p[my_class][i][j] >= 0.5):
                    if(use_color==1):
                        print(color.CYAN+"1 "+color.END, end='')
                    else:
                        print("1 ", end='')
                else:
                    print(f"0 ", end='')
            print(f"")
        print(f"")

def PerformEMAlgorithm(binary_x, rows, cols, EPSILON, termin_cnt, use_color, PSEUDO_CNST_LAMBDA, PSEUDO_CNST_P, pseudo_p_method, num_of_images, use_pseudo_w, use_pseudo_lambda, use_pseudo_p):
    count = 0
    #Initialize parameters: lambda_set, lambda_set[class], probability of being the class
    lambda_set = [random.uniform(0, 1) for x in range(10)]
    #lambda_set = [0.1 for x in range(10)]
    lambda_set = np.array([x/sum(lambda_set) for x in lambda_set]) #normalization

    #Initialize parameters: p[class][i][j] , probability of Bernoulli, initially random
    #p = np.array([[[random.uniform(0, 1) for j in range(28)] for i in range(28)] for i in range(10)])
    p = np.array([[[np.random.rand() for j in range(28)] for i in range(28)] for i in range(10)])

    while(True):
        #Calculate new W = lambda*(p^Xi)*((1-p)^(1-Xi))
        w_all_images            = E_Step(lambda_set, p, binary_x, rows, cols, num_of_images, use_pseudo_w)
        (new_lambda_set, new_p) = M_Step(w_all_images, p, binary_x, rows, cols, num_of_images, PSEUDO_CNST_LAMBDA, PSEUDO_CNST_P, pseudo_p_method, use_pseudo_lambda, use_pseudo_p)

        #Calculate difference
        diff = np.linalg.norm(np.array(new_p)-np.array(p)) + np.linalg.norm(np.array(new_lambda_set)-np.array(lambda_set))

        #Update parameters
        lambda_set = new_lambda_set
        p          = new_p

        #Visualize the current result
        DisplayImagination(p, use_color, rows, cols)
        print(f"No. of Iteration: {count+1}, Difference: {diff}")

        #Check terminating condition
        if((diff < EPSILON) or ((count+1) >= termin_cnt)):
            break;

        count += 1

    return (lambda_set, p, (count+1))

@nb.jit(nopython=True, nogil=True)
def CalculateTrueNegative(real_class, predict):
    result = 0
    for i in range(10):
        if(i==real_class):
            continue
        for j in range(10):
            if(j==real_class):
                continue
            result += predict[i][j]

    return result

@nb.jit(nopython=True, nogil=True)
def CalculateFalseNegative(real_class, predict):
    result = 0
    for i in range(10):
        if(i==real_class):
            continue
        for j in range(10):
            if(j!=real_class):
                continue
            result += predict[i][j]

    return result

@nb.jit(nopython=True, nogil=True)
def CalculateFalsePositive(real_class, predict):
    result = 0
    for i in range(10):
        if(i==real_class):
            continue
        result += predict[real_class][i]

    return result

def CalConfusionMatrixPrintResult(predict, num_of_images, total_iteration):
    total_tp = 0
    print(f"")
    print(f"")
    print(f"")

    for real_class in range(10):
        #Calculate confustion matrix
        true_positive  = predict[real_class][real_class]
        false_positive = CalculateFalsePositive(real_class, predict)
        false_negative = CalculateFalseNegative(real_class, predict)
        true_negative  = CalculateTrueNegative(real_class, predict)

        sensitivity = true_positive/(true_positive+false_negative)
        specificity = true_negative/(true_negative+false_positive)

        total_tp    += true_positive

        print(f"Confusion Matrix: {real_class}")
        print(f"\t\tPredict number {real_class}\tPredict not number {real_class}")
        print(f"Is    number {real_class} \t\t{true_positive}\t\t\t{false_negative}")
        print(f"Isn't number {real_class} \t\t{false_positive}\t\t\t{true_negative}")
        print(f"")
        print(f"Sensitivity (Successfully predict number {real_class})    : {sensitivity}")
        print(f"Specificity (Successfully predict not number {real_class}): {specificity}")
        print(f"")
        if(real_class != 9):
            print(f"--------------------------------------------")
            print(f"")

    error_rate = 1-(total_tp/num_of_images)
    print(f"Total iteration to converge: {total_iteration}")
    print(f"Total error rate: {error_rate:10.20f}")

@nb.jit(nopython=True, nogil=True)
def PredictResult(binary_x, train_y, lambda_set, p, num_of_images, rows, cols, mapping):
    predict   = np.zeros((10, 10))

    #Calculating the Frequency of calculated clustering to ground truth
    for index in range(num_of_images):
        #w = numba.typed.List([1 for x in range(10)])
        w = np.ones(10)

        for my_class in range(10):
            w[my_class] *= lambda_set[my_class]

            #Calculate all independent Bernouli(p) of each pixel in an image
            for i in range(rows):
                for j in range(cols):
                    if(binary_x[index][i][j]==1):
                        w[my_class] *= p[my_class][i][j]
                    else:
                        w[my_class] *= (1-p[my_class][i][j])

        #Find the max clustering
        w_max = w[0]
        max_index = 0
        for index_w in range(10):
            if(w_max < w[index_w]):
                w_max = w[index_w]
                max_index = index_w

        #Voting for most likely cluster
        predict[int(mapping[max_index])][train_y[index]] += 1

    return predict

@nb.jit(nopython=True, nogil=True)
def CalculateMapping(binary_x, train_y, lambda_set, p, num_of_images, rows, cols):
    mapping   = np.zeros(10)
    count     = np.zeros((10, 10))
    done_map  = 0
    cal_done  = np.zeros(10)
    true_done = np.zeros(10)

    #Calculating the Frequency of calculated clustering to ground truth
    for index in range(num_of_images):
        #w = numba.typed.List([1 for x in range(10)])
        w = np.ones(10)

        for my_class in range(10):
            w[my_class] *= lambda_set[my_class]

            #Calculate all independent Bernouli(p) of each pixel in an image
            for i in range(rows):
                for j in range(cols):
                    if(binary_x[index][i][j]==1):
                        w[my_class] *= p[my_class][i][j]
                    else:
                        w[my_class] *= (1-p[my_class][i][j])

        #Find the max clustering
        w_max = w[0]
        max_index = 0
        for index_w in range(10):
            if(w_max < w[index_w]):
                w_max = w[index_w]
                max_index = index_w

        #Voting for most likely cluster
        count[max_index][train_y[index]] += 1

    #Construct the mapping based on the count[cluster][true_label]
    while(done_map < 10):
        max_val         = -1
        max_cal_cluster = -1
        max_true_label  = -1

        for cal_cluster in range(10):
            if(cal_done[cal_cluster]==1):
                continue

            for true_label in range(10):
                if(true_done[true_label]==1):
                    continue

                if(count[cal_cluster][true_label] > max_val):
                    max_val               = count[cal_cluster][true_label]
                    max_cal_cluster       = cal_cluster
                    max_true_label        = true_label

        done_map += 1
        mapping[max_cal_cluster] = max_true_label
        cal_done[max_cal_cluster] = 1
        true_done[max_true_label] = 1

    return mapping, count


@nb.jit(nopython=True, nogil=True)
def E_Step(lambda_set, p, binary_x, rows, cols, num_of_images, use_pseudo_w):
    #w_all_images = numba.typed.List([[0 for x in range(10)] for y in range(len(binary_x))])
    w_all_images = np.zeros((num_of_images, 10))

    for index in range(num_of_images):
        #w = numba.typed.List([1 for x in range(10)])
        w = np.ones(10)

        for my_class in range(10):
            #print(f"lambda_set = {lambda_set}")
            w[my_class] *= lambda_set[my_class]

            #Calculate all independent Bernouli(p) of each pixel in an image
            for i in range(rows):
                for j in range(cols):
                    if(binary_x[index][i][j]==1):
                        #print(f"i = {i}, j = {j}, w[{my_class}] = {w[my_class]}, p[{my_class}][{i}][{j}] = {p[my_class][i][j]}", end='')
                        w[my_class] *= p[my_class][i][j]
                        #print(f", w[{my_class}] = {w[my_class]}")
                    else:
                        #print(f"i = {i}, j = {j}, w[{my_class}] = {w[my_class]}, 1-p[{my_class}][{i}][{j}] = {1-p[my_class][i][j]}", end='')
                        w[my_class] *= (1-p[my_class][i][j])
                        #print(f", w[{my_class}] = {w[my_class]}")
            #print(f'w[{my_class}] = {w[my_class]}')
            #print(f'------------------------------------')

        #Normalization
        sum_of_w = sum(w)
        if(sum_of_w != 0):
            for i in range(10):
                w[i] = w[i]/sum_of_w
        else: #every w is 0
            if(use_pseudo_w == 1):
                for i in range(10):
                    w[i] = 0.1

        #Assign final values
        w_all_images[index] = w

    return w_all_images

@nb.jit(nopython=True, nogil=True)
def M_Step(w_all_images, p, binary_x, rows, cols, num_of_images, PSEUDO_CNST_LAMBDA, PSEUDO_CNST_P, pseudo_p_method, use_pseudo_lambda, use_pseudo_p):
    #Update lambda_set
    lambda_sumup = np.zeros(10)
    for w_of_image in w_all_images:
        for index in range(10):
            lambda_sumup[index] += w_of_image[index]

    #--MLE lambda = sum(wi)/n
    #new_lambda_set = [x/num_of_images for x in lambda_sumup]
    new_lambda_set = np.zeros(10)
    for x in range(10):
        new_lambda_set[x] = lambda_sumup[x]/num_of_images

    '''
    print(f"use_pseudo_lambda  = {use_pseudo_lambda}")
    print(f"use_pseudo_p       = {use_pseudo_p}")
    print(f"PSEUDO_CNST_LAMBDA = {PSEUDO_CNST_LAMBDA}")
    print(f"PSEUDO_CNST_P      = {PSEUDO_CNST_P}")
    '''

    #Update p
    #new_p = [[[0 for j in range(28)] for i in range(28)] for i in range(10)]
    new_p = np.zeros((10, 28, 28))
    for my_class in range(10):
        got_zero_cond = 0
        for i in range(rows):
            for j in range(cols):
                for index in range(num_of_images):
                    new_p[my_class][i][j] += binary_x[index][i][j]*w_all_images[index][my_class]

                if(lambda_sumup[my_class] != 0):
                    new_p[my_class][i][j] /= lambda_sumup[my_class] #--MLE p = sum(wi*xi)/sum(wi)
                else:
                    if(use_pseudo_p == 1):
                        new_p[my_class][i][j] = PSEUDO_CNST_P
                    '''
                    got_zero_cond += 1
                    print(f"lambda_sumup[{my_class}] = {lambda_sumup[my_class]}")
                    print(f"new_p[{my_class}][{i}][{j}] = {new_p[my_class][i][j]}")
                    '''

        '''
        if(got_zero_cond >=1):
            print(f"lambda_sumup[{my_class}] = {lambda_sumup[my_class]}")
            for index in range(num_of_images):
                print(f"w_all_images[{index}][{my_class}] = {w_all_images[index][my_class]}")
            print(f"Before min calculation:")
            DisplayOneClass(new_p[my_class], my_class, 0, rows, cols)
            print(f"-------------->)")
            print(f"After min calculation:")
            DisplayOneClass(new_p[my_class], my_class, 0, rows, cols)
            sys.exit()
        '''

    if((sum(new_lambda_set) == 0) and (use_pseudo_lambda == 1)):
        for i in range(10):
            new_lambda_set[i] = PSEUDO_CNST_LAMBDA

    return (new_lambda_set, new_p)

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
        print(f"len(x_data) = {len(x_data)}")
#        print(f"xdata = ")
#        for i in range(len(x_data)):
#            print(f"i = {i}, {x_data[i]}")

    return x_data

def PrintMatrix(input_matrix, matrix_name):
    print(f'{len(input_matrix)}x{len(input_matrix[0])}, {matrix_name}: ')
#    print(f'[', end = '')
    for index_i, rows in enumerate(input_matrix):
        for index_j, cols in enumerate(rows):
            if(index_i == (len(input_matrix)-1) and index_j == (len(rows)-1)):
                print(f'{input_matrix[index_i][index_j]:20.1f}') #will print the same
                #print(f'[{cols}] ') #will print the same
            elif(index_j == (len(rows)-1)):
                print(f'{input_matrix[index_i][index_j]:20.1f}') #will print the same
            else:
                if(index_j == 0 and index_i != 0):
                    print(f'{input_matrix[index_i][index_j]:20.1f}', end='') #will print the same
                else:
                    print(f'{input_matrix[index_i][index_j]:20.1f}', end='') #will print the same

@nb.jit(nopython=True, nogil=True)
def Transform2Bins(train_x, train_y, is_debug, num_of_images, rows=28, cols=28):
    binary_x = np.zeros((num_of_images, rows, cols))
    for index in range(num_of_images):
        binary_x[index] = TallyFrequency2Bins(train_x[index], rows, cols)
        '''
        if(is_debug):
            print(f"index = {index}, label = {train_y[index]}")
            ShowImage(train_x[index])
            print(f"binary_x[index] = {binary_x[index]}")
        '''

    return binary_x

def StoreClassify(train_y, train_x, is_debug, rows, cols):
    all_ximage_storing = {} #storing 0-9 all corresponding train_x image
    all_xindex_storing = {} #storing 0-9 all corresponding train_x image original index
    all_disximage_storing = {}#storing 0-9 all corresponding train_x image that value in each pixelis in range of [0, 32] instead of original [0, 255]

    #Initialization
    for i in range(10):
        all_ximage_storing[i] = []
        all_xindex_storing[i] = []
        all_disximage_storing[i] = []

    #Classifying according to train_y value : 0-9
    for index, value in enumerate(train_y):
        all_ximage_storing[value].append(train_x[index])
        all_xindex_storing[value].append(index)

    for key, image_list in all_ximage_storing.items(): #key = 0-9 symbol, value = list of correspondent x_image
        for x_image in image_list:
            all_disximage_storing[key].append(TallyFrequency32Bins(x_image, rows, cols))

#    if(is_debug):
#        i = 7
#        print(f"i = {i}")
#        for index, x in enumerate(all_ximage_storing[i]):
#           print(f"the index of this image is {all_xindex_storing[i][index]}")
#           print(f"correspondent y is {train_y[all_xindex_storing[i][index]]}")
#           ShowImage(x)
#           ShowImage(all_disximage_storing[i][index])
#           print(f"-------------------------------------------------------------------")

    return (all_ximage_storing, all_disximage_storing, all_xindex_storing)

@nb.jit(nopython=True, nogil=True)
def TallyFrequency2Bins(x_image, rows=28, cols=28):
    x_discrete_image = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            x_discrete_image[i][j] = int(x_image[i][j]/128) # transform the gray level 0-255 to 0-1, total 2 bins.

    return x_discrete_image

def TallyFrequency32Bins(x_image, rows=28, cols=28):
    x_discrete_image = []

    for i in range(rows):
        row_val = []
        for j in range(cols):
            row_val.append(int(x_image[i][j]/8)) # transform the gray level 0-255 to 0-31, total 32 bins.
        x_discrete_image.append(row_val)

    return x_discrete_image

def CalculateImaginationContinuous(bin_map_stats, rows, cols):
    imagination_continuous = {} #0-9

    for label in range(10):
        ans = [[0 for j in range(cols)] for i in range(rows)]
        for i in range(rows):
            for j in range(cols):
                #Calculate the expectation
                if(bin_map_stats[label][i][j][0] >= 128):
                    ans[i][j] = 1
                else:
                    ans[i][j] = 0
        imagination_continuous[label] = ans

    return imagination_continuous

def CalculateImaginationDiscrete(bin_map_stats, all_disximage_storing, image_method_disc, rows, cols):
    imagination_discrete = {} #0-9

    for label in range(10):
        total_num_disx = len(all_disximage_storing[label])
        ans            = [[0 for j in range(cols)] for i in range(rows)]
        for i in range(rows):
            for j in range(cols):
                #Calculate the expectation
                if(image_method_disc==0):
                    #Use expectation mean to compare
                    if(ExpectationEstimation(bin_map_stats[label][i][j], total_num_disx) >= 128):
                        ans[i][j] = 1
                    else:
                        ans[i][j] = 0
                else:
                    #Use the white number and black number to compare
                    if(CompareLarge128NumberHigher(bin_map_stats[label][i][j], total_num_disx)):
                        ans[i][j] = 1
                    else:
                        ans[i][j] = 0

        imagination_discrete[label] = ans
    return imagination_discrete

def CompareLarge128NumberHigher(bin_range_list, total_num):
    high128_num = 0
    for bin_value, bin_num in enumerate(bin_range_list):
        if(bin_value >= 16):
            high128_num += bin_num

    return high128_num >= (total_num-high128_num)

def ExpectationEstimation(bin_range_list, total_num):
    exp_mean = 0
    for bin_value, bin_num in enumerate(bin_range_list):
        exp_mean += bin_value*bin_num

    exp_mean /= total_num
    exp_mean *= 8
    return exp_mean

def PrintImageMatrix(input_matrix, matrix_name):
    print(f'{matrix_name} = ')
    print(f'[', end = '')
    for index_i, rows in enumerate(input_matrix):
        for index_j, cols in enumerate(rows):
            if(index_i == (len(input_matrix)-1) and index_j == (len(rows)-1)):
                print(f'{input_matrix[index_i][index_j]}]') #will print the same
            elif(index_j == (len(rows)-1)):
                print(f'{input_matrix[index_i][index_j]} ') #will print the same
            else:
                if(index_j == 0 and index_i != 0):
                    print(f' {input_matrix[index_i][index_j]} ', end='') #will print the same
                else:
                    print(f'{input_matrix[index_i][index_j]} ', end='') #will print the same

def ShowImage(image_matrix):
    im = plt.imshow(image_matrix, cmap='gray', vmin=0, vmax=255)
    plt.show()

def PrintResult(test_y, posteriori_result, min_index_result, error_rate, toggle, imagination_result, use_color, rows=28, cols=28):
    if(toggle == 0):
        print(f"In Discrete Mode:")
    else:
        print(f"In Continuous Mode:")
    for index, label_y in enumerate(test_y):
        print(f"Postirior (in log scale)")
        for predict_label, posteriori in enumerate(posteriori_result[index]):
            print(f"{predict_label}: {posteriori}")
        print(f"Prediction: {min_index_result[index]}, Ans: {label_y}")
        print(f"")

    #Display the imageination of each digit in Naive Baye's Classifier.
    DisplayImagination(imagination_result, use_color, rows, cols)
    print(f"Error rate: {error_rate}")

#---------------Execution---------------#
if __name__ == '__main__':
    main()
