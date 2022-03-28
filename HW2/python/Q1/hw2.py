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
    (infile_train_label, infile_train_image, infile_test_label, infile_test_image, toggle, pseudo_cnt_method, PSEUDO_CNST, method, d_gauss, image_method_disc, use_color, is_debug) = ArgumentParser()

    #Get the input data point
    train_y = ReadLabelFile(infile_train_label, 2049, is_debug)
    train_x = ReadImageFile(infile_train_image, 2051, is_debug)
    test_y  = ReadLabelFile(infile_test_label, 2049, is_debug)
    test_x  = ReadImageFile(infile_test_image, 2051, is_debug)

    #Store all training image according to its y
    rows = len(test_x[0])
    cols = len(test_x[0][0])
    (all_ximage_storing, all_disximage_storing, all_xindex_storing) = StoreClassify(train_y, train_x, toggle, is_debug, rows, cols)

    #Training using all train_y, train_x using Naive Baye's Classifier method
    (prior_prob, min_map_stats, bin_map_stats, imagination_result) = TrainProcedure(all_ximage_storing, all_disximage_storing, toggle, image_method_disc, rows, cols)

    #Test each image in test_x with ground truth test_y
    (posteriori_result, min_index_result, error_rate) = TestProcedure(test_y, test_x, all_ximage_storing, all_disximage_storing, prior_prob, min_map_stats, bin_map_stats, toggle, pseudo_cnt_method, PSEUDO_CNST, rows, cols, method, d_gauss)

    #Print the result
    PrintResult(test_y, posteriori_result, min_index_result, error_rate, toggle, imagination_result, use_color, rows, cols)

    #Print the debug messages when necessary
    if(is_debug):
        pass
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

'''


#########################
#     Sub-Routine       #
#########################
def ArgumentParser():
    infile_train_label= None
    infile_train_image= None
    infile_test_label = None
    infile_test_image = None
    toggle            = None
    pseudo_cnt_method = None
    PSEUDO_CNST       = None
    method            = None
    d_gauss           = None
    image_method_disc = None
    use_color         = None
    is_debug          = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--infile_train_label", "-inf_tr_label", help="should set the input training label file.")
    parser.add_argument("--infile_train_image", "-inf_tr_image", help="should set the input training image file.")
    parser.add_argument("--infile_test_label", "-inf_tst_label", help="should set the input testing label file.")
    parser.add_argument("--infile_test_image", "-inf_tst_image", help="should set the input testing image file.")
    parser.add_argument("--toggle", "-tgl", help="Set '0' for discrete mode, '1' for continuous mode.")
    parser.add_argument("--pseudo_cnt_method", "-psc_method", help="Set '0' for finding min bin count that is not zero in the pixel position when encounting empty bin, '1' for directly setting the empty bin to the number of PSEUDO_CNST.")
    parser.add_argument("--PSEUDO_CNST", "-PSC_CNST", help="Set the PSEUDO_CNST, which is used when '--pseudo_cnt_method 1' is set.")
    parser.add_argument("--method", "-mthd", help="Set the method when calculating Gaussian probability in continuous mode. '0' for directly using Gaussian PDF. '1' for adding PDF in the range of [intensity-d_gauss, intensity+d_gauss].")
    parser.add_argument("--d_gauss", "-dg", help="Set the deviation for calculating Gaussian probabily at center x using Z-table.")
    parser.add_argument("--image_method_disc", "-im_mth_disc", help="Set '0' for using expectation on 32bins to calculate the mean value and use it to compare whether >=16 to print '1' or '0' otherwise. Set '1' for using whether number of bins in [0, 15] is larger than [16, 31], and to set it '0' or '1' otherwise.")
    parser.add_argument("--use_color", "-uc", help="Set 1 to use color for display the imagination of Naive Baye's Classifier. Set 0 to print in plain text.")
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
    if args.pseudo_cnt_method:
        pseudo_cnt_method = int(args.pseudo_cnt_method)
    if args.PSEUDO_CNST:
        PSEUDO_CNST = float(args.PSEUDO_CNST)
    if args.method:
        method = int(args.method)
    if args.d_gauss:
        d_gauss = float(args.d_gauss)
    if args.image_method_disc:
        image_method_disc = int(args.image_method_disc)
    if args.use_color:
        use_color = int(args.use_color)
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


    if(pseudo_cnt_method ==  None):
        print(f"Warning: You did not set --pseudo_cnt_method. The pseudo count when encountering empty bin will use the minimum of the number of other bin in the pixel")
        pseudo_cnt_method = 0

    if(pseudo_cnt_method != 0 and PSEUDO_CNST == None):
        print(f"Error: You choose to use PSEUDO_CNST as the number of pseudo count in your empty bin, but you did not set --PSEUDO_CNST.")
        sys.exit()

    if(method == None):
        method = 0

    if(d_gauss == None):
        d_gauss = 0.5

    if(image_method_disc == None and toggle == 0):
        print(f"Warning: You did not set the image_method_disc using '--image_method_disc', the image_method_disc will be set to 1.")
        image_method_disc = 1

    if(use_color == None):
        use_color = 1

    if(is_debug):
        print(f"infile_train_label = {infile_train_label}")
        print(f"infile_train_image = {infile_train_image}")
        print(f"infile_test_label  = {infile_test_label}")
        print(f"infile_test_image  = {infile_test_image}")
        print(f"toggle             = {toggle}")
        print(f"pseudo_cnt_method  = {pseudo_cnt_method}")
        print(f"PSEUDO_CNST        = {PSEUDO_CNST}")
        print(f"method             = {method}")
        print(f"d_gauss            = {d_gauss}")
        print(f"image_method_disc  = {image_method_disc}")
        print(f"use_color          = {use_color}")

    return (infile_train_label, infile_train_image, infile_test_label, infile_test_image, toggle, pseudo_cnt_method, PSEUDO_CNST, method, d_gauss, image_method_disc, use_color, is_debug)


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

def StoreClassify(train_y, train_x, toggle, is_debug, rows, cols):
    all_ximage_storing = {} #storing 0-9 all corresponding train_x image
    all_xindex_storing = {} #storing 0-9 all corresponding train_x image original index
    all_disximage_storing = {}#storing 0-9 all corresponding train_x image that value in each pixelis in range of [0, 32] instead of original [0, 255]

    #Initialization
    for i in range(10):
        all_ximage_storing[i] = []
        all_xindex_storing[i] = []
        if(toggle == 0):
            all_disximage_storing[i] = []

    #Classifying according to train_y value : 0-9
    for index, value in enumerate(train_y):
        all_ximage_storing[value].append(train_x[index])
        all_xindex_storing[value].append(index)

    if(toggle==0):
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


def GaussianPDF(x, mean, var):
    return ((1/math.sqrt(2*math.pi*var))*math.exp((-1/(2*var))*((x-mean)**2)))

def BoundaryCondition(intensity, max_val, min_val):
    if(intensity > max_val):
        intensity = max_val

    if(intensity < min_val):
        intensity = min_val

    return intensity


def GaussianProbabilityCal(test_ximage_intensity, mean, var, method, d_gauss):
    if(method == 0):
        #Take the -1*ln(prob) transform, where prob is the Gaussian probability
        prob = 0.5*(math.log(2*math.pi*var) + (1/var)*((test_ximage_intensity-mean)**2))
    elif(method == 1):
        prob  = 0
        for i in range(int(BoundaryCondition(test_ximage_intensity-d_gauss, 255, 0)), int(BoundaryCondition(test_ximage_intensity+d_gauss, 255, 0) + 1), 1):
            prob += GaussianPDF(i, mean, var)

        #Take the -1*ln(prob) transform, where prob is the Gaussian probability
        if(prob == 0):
            prob = 0.00000000001
        prob = -1*math.log(prob)

    return prob

def CalculateLikelihoodContinuous(target_y, test_ximage_intensity, i, j, min_map_stats, bin_map_stats, pseudo_cnt_method, PSEUDO_CNST, method, d_gauss):
    mean = bin_map_stats[target_y][i][j][0]
    var  = bin_map_stats[target_y][i][j][1]

    if(var == 0):
        if(pseudo_cnt_method == 0):
#            var = min_map_stats[target_y]
            return -1
        else:
            var = PSEUDO_CNST

    likelihood     = GaussianProbabilityCal(test_ximage_intensity, mean, var, method, d_gauss)
    return likelihood

def PredictCalculatedContinuous(target_y, test_ximage, prior_prob, min_map_stats, bin_map_stats, pseudo_cnt_method, PSEUDO_CNST, rows, cols, method, d_gauss):
    total_prob     = 0

    for i in range(rows):
        for j in range(cols):
            prob = CalculateLikelihoodContinuous(target_y, test_ximage[i][j], i, j, min_map_stats, bin_map_stats, pseudo_cnt_method, PSEUDO_CNST, method, d_gauss)

            if(prob == -1):
                continue

            total_prob += prob

    total_prob += -1*math.log(prior_prob[target_y])
    return total_prob

def CalculateLikelihoodDiscrete(target_y, test_ximage_intensity, all_disximage_storing, i, j, min_map_stats, bin_map_stats, pseudo_cnt_method, PSEUDO_CNST):
    positive_count = bin_map_stats[target_y][i][j][test_ximage_intensity]
    total_count    = len(all_disximage_storing[target_y])

    if(positive_count == 0):
        if(pseudo_cnt_method == 0):
            positive_count = min_map_stats[target_y][i][j]
        else:
            positive_count = PSEUDO_CNST

    likelihood     = positive_count/total_count
    return likelihood

def PredictCalculatedDiscrete(target_y, test_disc_ximage, all_disximage_storing, prior_prob, min_map_stats, bin_map_stats, pseudo_cnt_method, PSEUDO_CNST, rows=28, cols=28):
    total_prob     = 0

    for i in range(rows):
        for j in range(cols):
            prob = CalculateLikelihoodDiscrete(target_y, test_disc_ximage[i][j], all_disximage_storing, i, j, min_map_stats, bin_map_stats, pseudo_cnt_method, PSEUDO_CNST)
            total_prob += -1*math.log(prob)

    total_prob += -1*math.log(prior_prob[target_y])
    return total_prob

def CalculatePrior(all_ximage_storing):
    total_num = 0
    partial_sum = {}
    ans_sum_list  = []
    ans_prob_list = []

    for key, list_of_ximage in all_ximage_storing.items():
        partial_sum[key] = len(list_of_ximage)

    for i in range(10):
        ans_sum_list.append(partial_sum[i])

    total_num = sum(ans_sum_list)

    for i in range(10):
        ans_prob_list.append(partial_sum[i]/total_num)

    return ans_prob_list

def BinStatisticsEachDiscrete(all_disximage_storing, rows=28, cols=28):
    min_map_stats = {} #key: 0-9
    bin_map_stats = {} #key: 0-9

    for target, image_list in all_disximage_storing.items():
        bin_map = [[[0 for k in range(32)] for j in range(cols)] for i in range(rows)]
        min_map = [[-1 for j in range(cols)] for i in range(rows)]

        for image in image_list:
            for i in range(rows):
                for j in range(cols):
                    bin_map[i][j][image[i][j]] += 1

        for i in range(rows):
            for j in range(cols):
                #Find proper initial value
                for x in bin_map[i][j]:
                    if(x != 0):
                        min_val = x
                        break
                #Find Min
                for x in bin_map[i][j]:
                    if(x != 0 and x < min_val):
                        min_val = x

                min_map[i][j] = min_val

        min_map_stats[target] = min_map
        bin_map_stats[target] = bin_map

    return (min_map_stats, bin_map_stats)

def BinStatisticsEachContinuous(all_ximage_storing, rows=28, cols=28):
    min_map_stats = {} #key: 0-9
    bin_map_stats = {} #key: 0-9

    for target, image_list in all_ximage_storing.items():
        total_num_each_pixel = len(image_list)
        bin_map = [[[0 for k in range(2)] for j in range(cols)] for i in range(rows)]

        #mean calculation
        for image in image_list:
            for i in range(rows):
                for j in range(cols):
                    bin_map[i][j][0] += image[i][j]

        for i in range(rows):
            for j in range(cols):
                bin_map[i][j][0] /= total_num_each_pixel

        #variance calculation
        for image in image_list:
            for i in range(rows):
                for j in range(cols):
                    bin_map[i][j][1] += ((image[i][j]-bin_map[i][j][0])**2)

        for i in range(rows):
            for j in range(cols):
                bin_map[i][j][1] /= total_num_each_pixel

        #Find min variance within a series of ximage sharing with same target
        min_val = math.inf
        for i in range(rows):
            for j in range(cols):
                if(min_val > bin_map[i][j][1] and bin_map[i][j][1] != 0):
                    min_val = bin_map[i][j][1]

        min_map_stats[target] = min_val
        bin_map_stats[target] = bin_map

    return (min_map_stats, bin_map_stats)

def TrainProcedure(all_ximage_storing, all_disximage_storing, toggle, image_method_disc, rows=28, cols=28):
    prior_prob                                  = CalculatePrior(all_ximage_storing) #do not care whether to use all_ximage_storing or all_disximage_storing

    if(toggle == 0):
        (min_map_stats, bin_map_stats)          = BinStatisticsEachDiscrete(all_disximage_storing, rows, cols) # [10][28][28] each store the value of min bin that is non-zero
        (imagination_result)                    = CalculateImaginationDiscrete(bin_map_stats, all_disximage_storing, image_method_disc, rows, cols)
    else:
        (min_map_stats, bin_map_stats)          = BinStatisticsEachContinuous(all_ximage_storing, rows, cols) # [10][28][28] each store the value of min bin that is non-zero
        (imagination_result)                    = CalculateImaginationContinuous(bin_map_stats,rows, cols)

    return (prior_prob, min_map_stats, bin_map_stats, imagination_result)

def TestProcedure(test_y, test_x, all_ximage_storing, all_disximage_storing, prior_prob, min_map_stats, bin_map_stats, toggle, pseudo_cnt_method, PSEUDO_CNST, rows, cols, method, d_gauss):
    posteriori_result = {}
    min_index_result  = {}
    error             = 0
    error_rate        = 0

    for index, label_y in enumerate(test_y):
        (posteriori_ans, min_index) = PerformNaiveBayesClassifier(test_x[index], all_disximage_storing, prior_prob, min_map_stats, bin_map_stats, toggle, pseudo_cnt_method, PSEUDO_CNST, rows, cols, method, d_gauss)
        posteriori_result[index] = posteriori_ans
        min_index_result[index]  = min_index
#        if(index%10==0):
#            print(f"index = {index}, progress = {index/len(test_y)*100}%")

        if(min_index != label_y):
            error += 1

    error_rate = error/len(test_y)
    return posteriori_result, min_index_result, error_rate

def PerformNaiveBayesClassifier(test_ximage, all_disximage_storing, prior_prob, min_map_stats, bin_map_stats, toggle, pseudo_cnt_method, PSEUDO_CNST, rows, cols, method, d_gauss):
    posteriori_ans                          = [0 for i in range(10)] #posteriori of 0-9
    min_value                               = 0
    min_index                               = 0

    if(toggle == 0):
        #discrete mode
        test_disc_ximage = TallyFrequency32Bins(test_ximage, rows, cols)
        for target_y in range(10):
            posteriori_ans[target_y] = PredictCalculatedDiscrete(target_y, test_disc_ximage, all_disximage_storing, prior_prob, min_map_stats, bin_map_stats, pseudo_cnt_method, PSEUDO_CNST, rows, cols)

        #normalization
        total_sum      = sum(posteriori_ans)
        posteriori_ans = [x/total_sum for x in posteriori_ans]

        #find the min
        min_value = posteriori_ans[0]
        min_index = 0
        for index, value in enumerate(posteriori_ans):
            if( value < min_value):
                min_value = value
                min_index = index
    else:
        #continuous mode
        for target_y in range(10):
            posteriori_ans[target_y] = PredictCalculatedContinuous(target_y, test_ximage, prior_prob, min_map_stats, bin_map_stats, pseudo_cnt_method, PSEUDO_CNST, rows, cols, method, d_gauss)

        #normalization
        total_sum      = sum(posteriori_ans)
        posteriori_ans = [x/total_sum for x in posteriori_ans]

        #find the min
        min_value = posteriori_ans[0]
        min_index = 0
        for index, value in enumerate(posteriori_ans):
            if( value < min_value):
                min_value = value
                min_index = index

    return posteriori_ans, min_index

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

def DisplayImagination(imagination_result, use_color, rows=28, cols=28):
    print(f"Imagination of numbers in Bayesian classifier:")
    print(f"")

    for label in range(10):
        print(f"{label}: ")
        for i in range(rows):
            for j in range(cols):
                if(imagination_result[label][i][j] == 1):
                    if(use_color==1):
                        print(color.CYAN+"1 "+color.END, end='')
                    else:
                        print("1 ", end='')
                else:
                    print(f"0 ", end='')
            print(f"")
        print(f"")


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
