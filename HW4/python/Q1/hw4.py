#! /usr/bin/env python3
'''
    Author      : BCC
    Date        : 2022/03/31
'''

import argparse
import math
import sys
import re
import copy
import numpy as np
import gzip
import matplotlib.pyplot as plt
from scipy.special import expit
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
    (N, mx1, vx1, my1, vy1, mx2, vx2, my2, vy2, gaussian_meth, EPSILON_GRAD, EPSILON_NEWT, learning_rate, use_my_inverse, termin_cnt_grad, termin_cnt_newt, is_debug) = ArgumentParser()

    #Perform Logistic Regression
    print(f"> PerformLogisticRegression...")
    (design_matrix, Y, ground_truth_d1, ground_truth_d2, w_grad, w_newt) = PerformLogisticRegression(N, mx1, vx1, my1, vy1, mx2, vx2, my2, vy2, gaussian_meth, EPSILON_GRAD, EPSILON_NEWT, learning_rate, use_my_inverse, termin_cnt_grad, termin_cnt_newt, is_debug)

    #Print the result
    print(f"> PrintResult...")
    PrintResult(design_matrix, Y, ground_truth_d1, ground_truth_d2, w_grad, w_newt, N)

    if(is_debug):
        pass
        '''
        uni_gauss_point_x1 = UnivariateGaussianRandomGenerator(mx1, vx1, gaussian_meth)
        uni_gauss_point_y1 = UnivariateGaussianRandomGenerator(my1, vy1, gaussian_meth)
        uni_gauss_point_x2 = UnivariateGaussianRandomGenerator(mx2, vx2, gaussian_meth)
        uni_gauss_point_y2 = UnivariateGaussianRandomGenerator(my2, vy2, gaussian_meth)

        print(f"uni_gauss_point_x1 = {uni_gauss_point_x1}")
        print(f"uni_gauss_point_y1 = {uni_gauss_point_y1}")
        print(f"uni_gauss_point_x2 = {uni_gauss_point_x2}")
        print(f"uni_gauss_point_y2 = {uni_gauss_point_y2}")

        DrawGaussianDistribution(mx1, vx1, 1)
        DrawGaussianDistribution(my1, vy1, 1)
        DrawGaussianDistribution(mx2, vx2, 1)
        DrawGaussianDistribution(my2, vy2, 1)
        '''

#########################
#     Sub-Routine       #
#########################
def ArgumentParser():
    N                   = None
    mx1                 = None
    vx1                 = None
    my1                 = None
    vy1                 = None
    mx2                 = None
    vx2                 = None
    my2                 = None
    vy2                 = None
    gaussian_meth       = None
    EPSILON_GRAD        = pow(10, -4)
    EPSILON_NEWT        = pow(10, -4)
    learning_rate       = 1
    use_my_inverse      = 0
    termin_cnt_grad     = 200000
    termin_cnt_newt     = 200000
    is_debug            = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--N", "-N", help="The number of data points.")
    parser.add_argument("--mx1", "-mx1", help="The mean for univariate Gaussian data generator for point x1.")
    parser.add_argument("--vx1", "-vx1", help="The variance for univariate Gaussian data generator for point x1.")
    parser.add_argument("--my1", "-my1", help="The mean for univariate Gaussian data generator for point y1.")
    parser.add_argument("--vy1", "-vy1", help="The variance for univariate Gaussian data generator for point y1.")
    parser.add_argument("--mx2", "-mx2", help="The mean for univariate Gaussian data generator for point x2.")
    parser.add_argument("--vx2", "-vx2", help="The variance for univariate Gaussian data generator for point x2.")
    parser.add_argument("--my2", "-my2", help="The mean for univariate Gaussian data generator for point y2.")
    parser.add_argument("--vy2", "-vy2", help="The variance for univariate Gaussian data generator for point y2.")
    parser.add_argument("--gaussian_meth", "-gm", help="Set '0' to use 12 unform distribution to approximate standard Gaussian. Set '1' to use Box-Muller method to generate standard Gaussian. Default is '0'.")
    parser.add_argument("--epsilon_grad", "-eps_grad", help="set the error precision of the neighbored steps for terminating the Gradient Descent. Default 1e^-8.")
    parser.add_argument("--epsilon_newt", "-eps_newt", help="set the error precision of the neighbored steps for terminating the Newton's Method. Default 1e^-8.")
    parser.add_argument("--learning_rate", "-lr", help="It is the learning rate of Gradient Descent. Default 1.")
    parser.add_argument("--use_my_inverse", "-umi", help="Set 1 to use the inverse function by myself or set 0 to use numpy's. Default 0.")
    parser.add_argument("--termin_cnt_grad", "-tcg", help="Set the maximum iteration number for Gradient Descent. Default 200000.")
    parser.add_argument("--termin_cnt_newt", "-tcn", help="Set the maximum iteration number for Newton's Method. Default 200000.")
    parser.add_argument("--is_debug", "-isd", help="1 for debug mode; 0 for normal mode.")

    args = parser.parse_args()

    if(args.N):
        N   = int(args.N)
    if(args.mx1):
        mx1 = float(args.mx1)
    if(args.vx1):
        vx1 = float(args.vx1)
    if(args.my1):
        my1 = float(args.my1)
    if(args.vy1):
        vy1 = float(args.vy1)
    if(args.mx2):
        mx2 = float(args.mx2)
    if(args.vx2):
        vx2 = float(args.vx2)
    if(args.my2):
        my2 = float(args.my2)
    if(args.vy2):
        vy2 = float(args.vy2)
    if(args.gaussian_meth):
        gaussian_meth = int(args.gaussian_meth)
    if args.epsilon_grad:
        EPSILON_GRAD = float(args.epsilon_grad)
    if args.epsilon_newt:
        EPSILON_NEWT = float(args.epsilon_newt)
    if args.learning_rate:
        learning_rate = float(args.learning_rate)
    if args.use_my_inverse:
        use_my_inverse = bool(int(args.use_my_inverse))
    if args.termin_cnt_grad:
        termin_cnt_grad = int(args.termin_cnt_grad)
    if args.termin_cnt_newt:
        termin_cnt_newt = int(args.termin_cnt_newt)
    if(args.is_debug):
        is_debug = int(args.is_debug)

    if(N == None):
        print(f"Error: You should set '--N' or '-N' for the number of data points.")
        sys.exit()

    if(mx1 == None):
        print(f"Error: You should set '--mx1' or '-mx1' for the input of mean for univariate Gaussian data generator for point x1.")
        sys.exit()

    if(vx1 == None):
        print(f"Error: You should set '--vx1' or '-vx1' for the input of variance for univariate Gaussian data generator for point x1.")
        sys.exit()

    if(my1 == None):
        print(f"Error: You should set '--my1' or '-my1' for the input of mean for univariate Gaussian data generator for point y1.")
        sys.exit()

    if(vy1 == None):
        print(f"Error: You should set '--vy1' or '-vy1' for the input of variance for univariate Gaussian data generator for point x1.")
        sys.exit()

    if(mx2 == None):
        print(f"Error: You should set '--mx2' or '-mx2' for the input of mean for univariate Gaussian data generator for point x2.")
        sys.exit()

    if(vx2 == None):
        print(f"Error: You should set '--vx2' or '-vx2' for the input of variance for univariate Gaussian data generator for point x2.")
        sys.exit()

    if(my2 == None):
        print(f"Error: You should set '--my2' or '-my2' for the input of mean for univariate Gaussian data generator for point y2.")
        sys.exit()

    if(vy2 == None):
        print(f"Error: You should set '--vy2' or '-vy2' for the input of variance for univariate Gaussian data generator for point x2.")
        sys.exit()

    if(gaussian_meth == None):
        gaussian_meth = 0

    if(is_debug):
        print(f"N   = {N}")
        print(f"mx1 = {mx1}")
        print(f"vx1 = {vx1}")
        print(f"my1 = {my1}")
        print(f"vy1 = {vy1}")
        print(f"mx2 = {mx2}")
        print(f"vx2 = {vx2}")
        print(f"my2 = {my2}")
        print(f"vy2 = {vy2}")
        print(f"gaussian_meth   = {gaussian_meth}")
        print(f"EPSILON_GRAD    = {EPSILON_GRAD}")
        print(f"EPSILON_NEWT    = {EPSILON_NEWT}")
        print(f"learning_rate   = {learning_rate}")
        print(f"use_my_inverse  = {use_my_inverse}")
        print(f"termin_cnt_grad = {termin_cnt_grad}")
        print(f"termin_cnt_newt = {termin_cnt_newt}")

    return (N, mx1, vx1, my1, vy1, mx2, vx2, my2, vy2, gaussian_meth, EPSILON_GRAD, EPSILON_NEWT, learning_rate, use_my_inverse, termin_cnt_grad, termin_cnt_newt, is_debug)

def DrawGaussianDistribution(m, s, method):
    xaxis_range = np.arange(-10000, 10000, 1)
    yaxis_range = [UnivariateGaussianRandomGenerator(m, s, method) for x in xaxis_range]
    plt.hist(yaxis_range, 100)
    plt.title(f"Gaussian distribution using method {method}")
    plt.show()

def UnivariateGaussianRandomGenerator(m, s, method=0):
    data_point = 0.0

    if(method==0):
        data_point = (sum(np.random.uniform(0, 1, 12)) - 6) * math.sqrt(s) + m
    else:
        u_dist = np.random.uniform(0, 1)
        v_dist = np.random.uniform(0, 1)
        data_point = (math.sqrt((-2)*math.log(u_dist)) * math.sin(2*math.pi*v_dist)) * math.sqrt(s) + m

    return data_point


def PrintResult(design_matrix, Y, ground_truth_d1, ground_truth_d2, w_grad, w_newt, N):
    confusion_matrix_grad = [[0, 0], [0, 0]]
    confusion_matrix_newt = [[0, 0], [0, 0]]
    sensitivity_grad = 0
    specificity_grad = 0
    sensitivity_newt = 0
    specificity_newt = 0
    grad_d1 = []
    grad_d2 = []
    newt_d1 = []
    newt_d2 = []

    #First estimate the result of Gradient Descent
    for i in range(N*2):
        dot_result = MatrixMul([design_matrix[i]], w_grad)[0][0]
        if(expit(dot_result) >= 0.5):
            predict = 1
            grad_d2.append([design_matrix[i][0], design_matrix[i][1]])
        else:
            predict = 0
            grad_d1.append([design_matrix[i][0], design_matrix[i][1]])

        if(Y[i][0] == 0):
            if(predict == 0):
                confusion_matrix_grad[0][0] += 1
            else:
                confusion_matrix_grad[0][1] += 1
        else:
            if(predict == 0):
                confusion_matrix_grad[1][0] += 1
            else:
                confusion_matrix_grad[1][1] += 1


    sensitivity_grad = confusion_matrix_grad[0][0]/(confusion_matrix_grad[0][0]+confusion_matrix_grad[0][1])
    specificity_grad = confusion_matrix_grad[1][1]/(confusion_matrix_grad[1][0]+confusion_matrix_grad[1][1])

    #Second estimate the result of Newton's Method
    for i in range(N*2):
        dot_result = MatrixMul([design_matrix[i]], w_newt)[0][0]
        if(expit(dot_result) >= 0.5):
            predict = 1
            newt_d2.append([design_matrix[i][0], design_matrix[i][1]])
        else:
            predict = 0
            newt_d1.append([design_matrix[i][0], design_matrix[i][1]])

        if(Y[i][0] == 0):
            if(predict == 0):
                confusion_matrix_newt[0][0] += 1
            else:
                confusion_matrix_newt[0][1] += 1
        else:
            if(predict == 0):
                confusion_matrix_newt[1][0] += 1
            else:
                confusion_matrix_newt[1][1] += 1

    sensitivity_newt = confusion_matrix_newt[0][0]/(confusion_matrix_newt[0][0]+confusion_matrix_newt[0][1])
    specificity_newt = confusion_matrix_newt[1][1]/(confusion_matrix_newt[1][0]+confusion_matrix_newt[1][1])

    print(f"Gradient descent:")
    print(f"")
    PrintMatrix(w_grad, 'w')
    print(f"")
    print(f"Confusion Matrix:")
    print(f"\t\tPredict cluster 1\tPredict cluster 2")
    print(f"Is cluster 1 \t\t{confusion_matrix_grad[0][0]}\t\t\t{confusion_matrix_grad[0][1]}")
    print(f"Is cluster 2 \t\t{confusion_matrix_grad[1][0]}\t\t\t{confusion_matrix_grad[1][1]}")
    print(f"")
    print(f"Sensitivity (Successfully predict cluster 1): {sensitivity_grad}")
    print(f"Specificity (Successfully predict cluster 2): {specificity_grad}")
    print(f"--------------------------------------------")
    print(f"Newton's method:")
    print(f"")
    PrintMatrix(w_newt, 'w')
    print(f"")
    print(f"Confusion Matrix:")
    print(f"\t\tPredict cluster 1\tPredict cluster 2")
    print(f"Is cluster 1 \t\t{confusion_matrix_newt[0][0]}\t\t\t{confusion_matrix_newt[0][1]}")
    print(f"Is cluster 2 \t\t{confusion_matrix_newt[1][0]}\t\t\t{confusion_matrix_newt[1][1]}")
    print(f"")
    print(f"Sensitivity (Successfully predict cluster 1): {sensitivity_newt}")
    print(f"Specificity (Successfully predict cluster 2): {specificity_newt}")

    Visualization(ground_truth_d1, ground_truth_d2, grad_d1, grad_d2, newt_d1, newt_d2)

def Visualization(ground_truth_d1, ground_truth_d2, grad_d1, grad_d2, newt_d1, newt_d2):
    #creat the subplot object
    fig=plt.subplots(1,3)

    #======================Ground truth========================#
    plt.subplot(1, 3, 1)
    #plt.xlim(-5, 15)
    #plt.ylim(-3, 15)

    #setting the scale for separating x, y
    #x_major_locator=MultipleLocator(10)
    #y_major_locator=MultipleLocator(2)
    #ax=plt.gca()
    #ax.xaxis.set_major_locator(x_major_locator)
    #ax.yaxis.set_major_locator(y_major_locator)

    plt.title("Ground truth")
    ground_truth_d1x = [x[0] for x in ground_truth_d1]
    ground_truth_d1y = [x[1] for x in ground_truth_d1]
    ground_truth_d2x = [x[0] for x in ground_truth_d2]
    ground_truth_d2y = [x[1] for x in ground_truth_d2]
    if((len(ground_truth_d1x) !=0) and (len(ground_truth_d1y) !=0)):
        plt.scatter(ground_truth_d1x, ground_truth_d1y, c='red')
    if((len(ground_truth_d2x) !=0) and (len(ground_truth_d2y) !=0)):
        plt.scatter(ground_truth_d2x, ground_truth_d2y, c='blue')


    #======================Gradient Descent predict result========================#
    plt.subplot(1, 3, 2)
    #plt.xlim(-5, 15)
    #plt.ylim(-3, 15)

    #setting the scale for separating x, y
    #x_major_locator=MultipleLocator(10)
    #y_major_locator=MultipleLocator(2)
    #ax=plt.gca()
    #ax.xaxis.set_major_locator(x_major_locator)
    #ax.yaxis.set_major_locator(y_major_locator)

    plt.title("Gradient descent")
    grad_d1x = [x[0] for x in grad_d1]
    grad_d1y = [x[1] for x in grad_d1]
    grad_d2x = [x[0] for x in grad_d2]
    grad_d2y = [x[1] for x in grad_d2]
    if((len(grad_d1x) !=0) and (len(grad_d1y) !=0)):
        plt.scatter(grad_d1x, grad_d1y, c='red')
    if((len(grad_d2x) !=0) and (len(grad_d2y) !=0)):
        plt.scatter(grad_d2x, grad_d2y, c='blue')

    #======================Newton's Method predict result========================#
    plt.subplot(1, 3, 3)
    #plt.xlim(-5, 15)
    #plt.ylim(-3, 15)

    #setting the scale for separating x, y
    #x_major_locator=MultipleLocator(10)
    #y_major_locator=MultipleLocator(2)
    #ax=plt.gca()
    #ax.xaxis.set_major_locator(x_major_locator)
    #ax.yaxis.set_major_locator(y_major_locator)

    plt.title("Newton's method")
    newt_d1x = [x[0] for x in newt_d1]
    newt_d1y = [x[1] for x in newt_d1]
    newt_d2x = [x[0] for x in newt_d2]
    newt_d2y = [x[1] for x in newt_d2]
    if((len(newt_d1x) !=0) and (len(newt_d1y) !=0)):
        plt.scatter(newt_d1x, newt_d1y, c='red')
    if((len(newt_d2x) !=0) and (len(newt_d2y) !=0)):
        plt.scatter(newt_d2x, newt_d2y, c='blue')

    #show the plot
    plt.tight_layout()
    plt.show()

def PerformLogisticRegression(N, mx1, vx1, my1, vy1, mx2, vx2, my2, vy2, gaussian_meth, EPSILON_GRAD, EPSILON_NEWT, learning_rate, use_my_inverse, termin_cnt_grad, termin_cnt_newt, is_debug):
    #Form the design_matrix, and Y
    (design_matrix, Y, ground_truth_d1, ground_truth_d2) = FormDesignMatrix(N, mx1, vx1, my1, vy1, mx2, vx2, my2, vy2, gaussian_meth)
    #
    w_grad = GradientDescent(design_matrix, Y, EPSILON_GRAD, learning_rate, termin_cnt_grad, is_debug)
    w_newt = NewtonMethod(design_matrix, Y, EPSILON_NEWT, learning_rate, use_my_inverse, termin_cnt_newt, is_debug)

    return (design_matrix, Y, ground_truth_d1, ground_truth_d2, w_grad, w_newt)

def NewtonMethod(design_matrix, Y, EPSILON_NEWT, learning_rate, use_my_inverse, termin_cnt_newt, is_debug):
    w_newt      = [[0], [0], [0]] #initial
    count       = 0

    while(True):
        gradient_f                   = FormGradientMatrix(design_matrix, w_newt)
        gradient_result              = MatrixMul(MatrixTranspose(design_matrix), MatrixSub(gradient_f, Y))
        matrixD                      = FormDiagonal(design_matrix, w_newt)
        hessian_matrix               = MatrixMul(MatrixMul(MatrixTranspose(design_matrix), matrixD), design_matrix)
        hessian_inverse, has_inverse = MatrixInverse(hessian_matrix, use_my_inverse)
        if(has_inverse):
            w_newt_new = MatrixSub(w_newt, MatrixMul(hessian_inverse, gradient_result))
        else:
            w_newt_new = MatrixSub(w_newt, MatrixMulScalar(gradient_result, learning_rate))

        error_try      = ErrorVectorCalculation(MatrixSub(w_newt_new, w_newt))
        w_newt         = copy.deepcopy(w_newt_new)
        if(is_debug):
            print(f"count = {count}, newt")
            print(f"error_try = {error_try}, newt")

        if(error_try < EPSILON_NEWT or count > termin_cnt_newt):
            if(is_debug):
                if(error_try < EPSILON_NEWT):
                    print(f"Newton's Method: Terminates for error < {EPSILON_NEWT}")
                else:
                    print(f"Newton's Method: Terminates for count > {termin_cnt_newt}")
            break;

        count += 1

    return w_newt

def FormDiagonal(design_matrix, w_newt):
    matrixD    = []
    num        = len(design_matrix)

    for i in range(num):
        row = []
        dot_result = MatrixMul([design_matrix[i]], w_newt)[0][0]
        for j in range(num):
            if(i==j):
                #If dot_result is smaller than -700, it is no difference to set it -700.
                try:
                    row.append(math.exp(-1*dot_result)*math.pow(expit(dot_result), 2))
                except OverflowError:
                    if(dot_result < -700):
                        dot_result = -700
                    row.append(math.exp(-1*dot_result)*math.pow(expit(dot_result), 2))
                except TypeError:
                    dot_result = 10000000000000000
                    row.append(math.exp(-1*dot_result)*math.pow(expit(dot_result), 2))
            else:
                row.append(0)
        matrixD.append(row)

    return matrixD

def GradientDescent(design_matrix, Y, EPSILON_GRAD, learning_rate, termin_cnt_grad, is_debug):
    w_grad      = [[0], [0], [0]] #initial
    count       = 0

    while(True):
        gradient_f      = FormGradientMatrix(design_matrix, w_grad)
        gradient_result = MatrixMul(MatrixTranspose(design_matrix), MatrixSub(gradient_f, Y))
        w_grad_new      = MatrixSub(w_grad, MatrixMulScalar(gradient_result, learning_rate))

        error_try       = ErrorVectorCalculation(MatrixSub(w_grad_new, w_grad))
        w_grad          = copy.deepcopy(w_grad_new)

        if(is_debug):
            print(f"count = {count}, grad")
            print(f"error_try = {error_try}, grad")
        if(error_try < EPSILON_GRAD or count > termin_cnt_grad):
            if(is_debug):
                if(error_try < EPSILON_GRAD):
                    print(f"Gradient Descent: Terminates for error < {EPSILON_GRAD}")
                else:
                    print(f"Gradient Descent: Terminates for count > {termin_cnt_grad}")

            break;

        count += 1

    return w_grad

def FormGradientMatrix(design_matrix, w_grad):
    gradient_f = []
    num        = len(design_matrix)

    for i in range(num):
        dot_result = MatrixMul([design_matrix[i]], w_grad)[0][0]
        try:
            gradient_f.append([expit(dot_result)])
        except TypeError:
            if(dot_result > 0):
                dot_result =  10000000000000000
            else:
                dot_result = -10000000000000000

            gradient_f.append([expit(dot_result)])

    return gradient_f

def ErrorVectorCalculation(vector_x):
    return (math.sqrt(MatrixMul(MatrixTranspose(vector_x), vector_x)[0][0])) # sqrt(||x||^2)

def MatrixInverse(matrix_a, use_my_inverse):
    has_inverse = 1

    if(not use_my_inverse):
        #print(f"Use NP inverse...")
        try:
            matrix_c = np.linalg.inv(np.array(matrix_a))
            matrix_c.tolist()
            x = matrix_c
            has_inverse = 1
        except:
            x = []
            has_inverse = 0
    else:
        #print(f"Use My inverse...")
        row_a = len(matrix_a)
        col_a = len(matrix_a[0])
        y     = []
        x     = [] #The inverse matrix

        if(row_a != col_a):
            print(f"Error: Input matrix: {matrix_a} is not a square matrix, and cannot be found an inverse through LU Decomposition.")
            sys.exit()
        if(DeterminationCal(matrix_a, row_a, col_a) == 0):
            #print("DeterminationCal = 0!")
            has_inverse = 0
            return x, has_inverse
            #print(f"Error: The det of matrix: {matrix_a} is 0, so it is not invertible.")
            #sys.exit()


        #Initialization of y and x
        for i in range(row_a):
            row_y = []
            row_x = []
            for j in range(col_a):
                row_y.append(0)
                row_x.append(0)

            y.append(row_y)
            x.append(row_x)


        #Perform the LU Decomposition
        (matrix_a_l, matrix_a_u, has_lu) = LUDecomposition(matrix_a)
        if(not has_lu):
            #print("No LU Decomposition!")
            has_inverse = 0
            return x, has_inverse

        #Perform the inverse - first step  : find the y of Ly = [e1 e2 e3], where e1, e2, e3 are columns of I
        for i in range(row_a):
            for j in range(col_a):
                if(i==0):
                    if(j==0):
                        y[i][j] = 1
                    else:
                        y[i][j] = 0
                else:
                    if(i==j):
                        tmp_y = 0
                        for k in range(i):
                            tmp_y += matrix_a_l[i][k]*y[k][j]
                        y[i][j] = 1 - tmp_y
                    else:
                        tmp_y = 0
                        for k in range(i):
                            tmp_y += matrix_a_l[i][k]*y[k][j]
                        y[i][j] = -tmp_y

        #Perform the inverse - second step : find the x of Ux = [y1 y2 y3], where y1, y2, y3 are columns of y
        for i in range((row_a-1), -1, -1):
            for j in range(col_a):
                if(i == (row_a-1)):
                    if(matrix_a_u[i][i] == 0):
                        has_inverse = 0
                        return x, has_inverse
                    x[i][j] = y[i][j]/matrix_a_u[i][i]
                else:
                    tmp_x = 0
                    for k in range((row_a-1), i, -1):
                        tmp_x += x[k][j]*matrix_a_u[i][k]

                    if(matrix_a_u[i][i] == 0):
                        has_inverse = 0
                        return x, has_inverse
                    x[i][j] = (y[i][j]-tmp_x)/matrix_a_u[i][i]

    return x, has_inverse

def PrintMatrix(input_matrix, matrix_name):
    print(f'{len(input_matrix)}x{len(input_matrix[0])}, {matrix_name}: ')
#    print(f'[', end = '')
    for index_i, rows in enumerate(input_matrix):
        for index_j, cols in enumerate(rows):
            if(index_i == (len(input_matrix)-1) and index_j == (len(rows)-1)):
                print(f'{input_matrix[index_i][index_j]:20.10f}') #will print the same
                #print(f'[{cols}] ') #will print the same
            elif(index_j == (len(rows)-1)):
                print(f'{input_matrix[index_i][index_j]:20.10f}') #will print the same
            else:
                if(index_j == 0 and index_i != 0):
                    print(f'{input_matrix[index_i][index_j]:20.10f}', end='') #will print the same
                else:
                    print(f'{input_matrix[index_i][index_j]:20.10f}', end='') #will print the same

def MatrixMulScalar(matrix_a, scalar=1):
    row_a = len(matrix_a)
    col_a = len(matrix_a[0])
    matrix_c = []

    #Initialization matrix_c
    for i in range(row_a):
        row = []
        for j in range(col_a):
            row.append(0)

        matrix_c.append(row)

    #Multiply every element with the scalar
    for i in range(row_a):
        for j in range(col_a):
            matrix_c[i][j] = scalar*matrix_a[i][j]

    return matrix_c

def MatrixIdentityGen(dimension, lamb=1):#lamb will multiply with 1 on the diagonal elements
    matrix_c = []

    for i in range(dimension):
        row = []
        for j in range(dimension):
            if(i==j):
                row.append(1*lamb)
            else:
                row.append(0)

        matrix_c.append(row)

    return matrix_c

def MatrixTranspose(matrix_a):
    row_a = len(matrix_a)
    col_a = len(matrix_a[0])
    matrix_c = []

    #Initialization matrix_c
    for i in range(col_a):
        row = []
        for j in range(row_a):
            row.append(0)

        matrix_c.append(row)

    for i in range(row_a):
        for j in range(col_a):
            matrix_c[j][i] = matrix_a[i][j]

    return matrix_c

def FormDesignMatrix(N, mx1, vx1, my1, vy1, mx2, vx2, my2, vy2, gaussian_meth):
    design_matrix   = []
    Y               = []
    ground_truth_d1 = []
    ground_truth_d2 = []

    #Class D1: label 0
    for i in range(N):
        row = []
        dx1 = UnivariateGaussianRandomGenerator(mx1, vx1, gaussian_meth)
        dy1 = UnivariateGaussianRandomGenerator(my1, vy1, gaussian_meth)
        row.append(dx1)
        row.append(dy1)
        row.append(1)
        ground_truth_d1.append([dx1, dy1])
        design_matrix.append(row)
        Y.append([0])

    #Class D1: label 1
    for i in range(N):
        row = []
        dx2 = UnivariateGaussianRandomGenerator(mx2, vx2, gaussian_meth)
        dy2 = UnivariateGaussianRandomGenerator(my2, vy2, gaussian_meth)
        row.append(dx2)
        row.append(dy2)
        row.append(1)
        ground_truth_d2.append([dx2, dy2])
        design_matrix.append(row)
        Y.append([1])

    return(design_matrix, Y, ground_truth_d1, ground_truth_d2)




def FormMatrix(input_data, poly_num):
    design_matrix = []
    b             = []

    for data_pairs in input_data:
        row_val = []
        for index in range(poly_num):
            row_val.append(pow(data_pairs[0], index))

        design_matrix.append(row_val)
        b.append([data_pairs[1]])

    return(design_matrix, b)

def UnivariateGaussianRandomGenerator(m, s, method=0):
    data_point = 0.0

    if(method==0):
        data_point = (sum(np.random.uniform(0, 1, 12)) - 6) * math.sqrt(s) + m
    else:
        u_dist = np.random.uniform(0, 1)
        v_dist = np.random.uniform(0, 1)
        data_point = (math.sqrt((-2)*math.log(u_dist)) * math.sin(2*math.pi*v_dist)) * math.sqrt(s) + m

    return data_point

def PolynomialBasisRandomGenerator(n, w, a, gaussian_meth):
    x = np.random.uniform(-1, 1)
    y = 0
    tmp_x = 1
    for i in range(n):
        if(i == 0):
            y += w[i]
        else:
            tmp_x *= x
            y += w[i]*tmp_x

    y += UnivariateGaussianRandomGenerator(0, a, gaussian_meth)
    return (x, y)

def ConvertToList(w_string):
    return (list(map(float, w_string.strip('[]').split(','))))

def MatrixSub(matrix_a, matrix_b):
    row_a = len(matrix_a)
    col_a = len(matrix_a[0])
    row_b = len(matrix_b)
    col_b = len(matrix_b[0])
    matrix_c = []

    if((row_a != row_b) or (col_a != col_b)):
        print(f"Error: Dimensions of matrix_a and matrix_b are different, and cannot be added together.")
        sys.exit()

    #Initialization matrix_c
    for i in range(row_a):
        row = []
        for j in range(col_a):
            row.append(0)

        matrix_c.append(row)

    #calculatations in matrix_c
    for i in range(row_a):
        for j in range(col_a):
            matrix_c[i][j] = matrix_a[i][j]-matrix_b[i][j]

    return matrix_c

def MatrixAdd(matrix_a, matrix_b):
    row_a = len(matrix_a)
    col_a = len(matrix_a[0])
    row_b = len(matrix_b)
    col_b = len(matrix_b[0])
    matrix_c = []

    if((row_a != row_b) or (col_a != col_b)):
        print(f"Error: Dimensions of matrix_a and matrix_b are different, and cannot be substracted together.")
        sys.exit()

    #Initialization matrix_c
    for i in range(row_a):
        row = []
        for j in range(col_a):
            row.append(0)

        matrix_c.append(row)

    #calculatations in matrix_c
    for i in range(row_a):
        for j in range(col_a):
            matrix_c[i][j] = matrix_a[i][j]+matrix_b[i][j]

    return matrix_c


def MatrixMul(matrix_a, matrix_b):
    row_a = len(matrix_a)
    col_a = len(matrix_a[0])
    row_b = len(matrix_b)
    col_b = len(matrix_b[0])

    if(col_a != row_b):
        print(f"Error: The number of columns of matrix_a is different with the number of rows of matrix_b, and ")
        print(f"Error: they cannot be multiplied together.")
        sys.exit()

    matrix_c = []
    row_c = row_a
    col_c = col_b

    #Initialization matrix_c
    for i in range(row_c):
        row = []
        for j in range(col_c):
            row.append(0)

        matrix_c.append(row)

    #calculatations in matrix_c
    for i in range(row_c):
        for j in range(col_c):
            tmp_result = 0
            for k in range(col_a):
                tmp_result += matrix_a[i][k]*matrix_b[k][j]

            matrix_c[i][j] = tmp_result

    return matrix_c

def SwapRow(matrix_a, row_i, row_j, num_row_a, num_col_a):
    if(row_i >= num_row_a or row_j >= num_col_a):
        print(f"Error: The input row_i or row_j is larger than the dimenssion of input matrix_a.")
        sys.exit()

    tmp_row         = copy.deepcopy(matrix_a[row_i])
    matrix_a[row_i] = copy.deepcopy(matrix_a[row_j])
    matrix_a[row_j] = copy.deepcopy(tmp_row)


def DeterminationCal(matrix_a, row_a=None, col_a=None):
    if(row_a == None):
        row_a = len(matrix_a)
    if(col_a == None):
        col_a = len(matrix_a[0])

    if(row_a != col_a):
        print(f"Error: The row and col of matrix_a are not identical, thus input metrix_a does not have det().")
        sys.exit()

    matrix_a_copy            = copy.deepcopy(matrix_a)
    have_found_row_to_change = 1
    det_result               = 0
    minus_tmp                = 1 #remember for the sign change owing to row exchange.
    for i in range(row_a):
        if(matrix_a_copy[i][i] == 0):
            have_found_row_to_change = 0
            for redund in range(i+1, row_a):
                if(matrix_a_copy[redund][i] != 0):
                    #Do the row exchange, and the result*-1
                    SwapRow(matrix_a_copy, i, redund, row_a, col_a)
                    have_found_row_to_change  = 1
                    minus_tmp                *= -1
                    break
        if(have_found_row_to_change == 0):
            det_result = 0
            break

        #Perform the row manipulation
        for redund in range(i+1, row_a):
            cancel_scalar = matrix_a_copy[redund][i]/matrix_a_copy[i][i]
            for j in range(i, col_a):
                matrix_a_copy[redund][j] = matrix_a_copy[redund][j] - cancel_scalar*matrix_a_copy[i][j]

    det_result = 1
    for i in range(row_a):
        det_result *= matrix_a_copy[i][i]
    det_result *= minus_tmp

    return det_result

def LeadingPrincipalSubmatrixFormation(matrix_a, n):
    sub_matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(matrix_a[i][j])

        sub_matrix.append(row)

    return sub_matrix


def LeadingPrincipalMinorCheck(matrix_a):
    row_a = len(matrix_a)
    col_a = len(matrix_a[0])
    det_has_zero = False
    nsub         = -1
    for i in range(row_a):
        sub_matrix = LeadingPrincipalSubmatrixFormation(matrix_a, i+1)
        det_result = DeterminationCal(sub_matrix)

        if(det_result == 0):
            det_has_zero = True
            nsub         = i+1
            break

    return (det_has_zero, nsub)


def LUDecomposition(matrix_a):
    row_a = len(matrix_a)
    col_a = len(matrix_a[0])
    matrix_l = []
    matrix_u = []

    #Check all leading principal submatrix has non-zero det.
    (det_has_zero, nsub) = LeadingPrincipalMinorCheck(matrix_a)
    if(det_has_zero):
        return(matrix_l, matrix_u, 0)
        #print(f"Error: The input matrix: {matrix_a} has 0 det in the {nsub} level of its leading principal submatrix.")
        #print(f"Error: Thus, this matrix has no LU Decomposition, and the inverse by LU Decomposition fails.")
        #sys.exit()


    #Initialization matrix_l, metrix_u
    for i in range(row_a):
        row1 = []
        row2 = []
        for j in range(col_a):
            row1.append(0)
            row2.append(0)

        matrix_l.append(row1)
        matrix_u.append(row2)

    #Perform the Doolittle Algorithm
    for i in range(row_a):
        for j in range(col_a):
            #Upper Triangle
            if(i==0):
                matrix_u[i][j] = matrix_a[i][j]
            elif(i>j):
                matrix_u[i][j] = 0
            else:
                tmp_for_u = 0
                for k in range(i):
                    tmp_for_u += matrix_l[i][k]*matrix_u[k][j]

                matrix_u[i][j] = matrix_a[i][j] - tmp_for_u

            #Lower Triangle
            if(j==0):
                matrix_l[i][j] = matrix_a[i][j]/matrix_u[j][j]
            elif(j==i):
                matrix_l[i][j] = 1
            elif(j>i):
                matrix_l[i][j] = 0
            else:
                tmp_for_l = 0
                for k in range(j):
                    tmp_for_l += matrix_l[i][k]*matrix_u[k][j]

                matrix_l[i][j] = (matrix_a[i][j]-tmp_for_l)/matrix_u[j][j]

    return (matrix_l, matrix_u, 1)

#---------------Execution---------------#
if __name__ == '__main__':
    main()
