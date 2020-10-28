# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 14:15:45 2020

@author: xzye0
"""
from svmutil import *
import matplotlib.pyplot as plt
import numpy as np

#Problem 2 & 3
#scaling process
y, x = svm_read_problem('abalone_formated.csv', return_scipy=True)
scale_para = csr_find_scale_param(x[:3133])
scaled_x = csr_scale(x, scale_para)

#Problem 4
def compute_cv_error(c, d):
    prob = svm_problem(y[:3133], x[:3133])
    param = svm_parameter('-s 0 -c {} -t 1 -d {} -v 10'.format(c, d))
    ACC = svm_train(prob, param)
    return 1 - ACC/100

def train(c, d):
    prob = svm_problem(y[:3133], x[:3133])
    param = svm_parameter('-s 0 -c {} -t 1 -d {}'.format(c, d))
    m = svm_train(prob, param)
    return m

def predict(m):
    p_label, p_acc, p_val = svm_predict(y[3133:], x[3133:], m)
    return p_label, p_acc, p_val, m.l

def p4_plot_cv_with_sig_degree(d):
    k = 4
    nb_exp = 20
    M = np.zeros(shape=(nb_exp, 2*k+1))
    
    x_axis = np.array([2**i for i in range(-1*k, k+1)])
    
    for i in range(nb_exp):
        for j in range(len(x_axis)):
            M[i][j] = compute_cv_error(x_axis[j], d)
    
    y_mean = np.mean(M, axis=0,dtype=np.float64)*100
    y_std = np.std(M, axis=0,dtype=np.float64)*100
    y_mean_plus = (y_mean + y_std)
    y_mean_minus = (y_mean - y_std)
    
    plt.plot(x_axis, y_mean, 'k-', label = "mean")
    plt.plot(x_axis, y_mean_plus, 'r--', label = "mean + std")
    plt.plot(x_axis, y_mean_minus, 'C1--',label = "mean - std")
    plt.title("Cross-validation error plot with degree %s" % d)
    plt.xlabel("$C = 2^{-k}$")
    plt.ylabel("error(%)")
    plt.legend()
    plt.savefig("degree_%s.png" % d)
    plt.show()

def p5_plot_cv_with_fixed_c():
    C = 16
    nb_exp = 20
    M = np.zeros(shape=(nb_exp, 4))
    
    x_axis = np.array([1,2,3,4])
    for i in range(nb_exp):
        for j in range(len(x_axis)):
            M[i][j] = compute_cv_error(C, x_axis[j])
    
    y_mean = np.mean(M, axis=0,dtype=np.float64)*100
    y_std = np.std(M, axis=0,dtype=np.float64)*100
    y_mean_plus = (y_mean + y_std)
    y_mean_minus = (y_mean - y_std)

    plt.plot(x_axis, y_mean, 'k-', label = "mean")
    plt.plot(x_axis, y_mean_plus, 'r--', label = "mean + std")
    plt.plot(x_axis, y_mean_minus, 'C1--',label = "mean - std")
    plt.title("Cross-validation error over different d")
    plt.xlabel("$d$")
    plt.xticks(range(1,5))
    plt.ylabel("error(%)")
    plt.legend()
    #plt.savefig("degree_%s.png")
    plt.show()
    
def p5_plot_test():
    C = 16
    
    x_axis = np.array([1,2,3,4])
    test_error = np.zeros(4)
    nb_sv = np.zeros(4)
    for i in range(len(x_axis)):
        model = train(C, x_axis[i])
        p = predict(model)
        test_error[i] = p[1][1] * 100
        nb_sv[i] = p[3]
    
    plt.plot(x_axis, test_error, 'k', label = "test error")
    plt.title("Test error over different d")
    plt.xlabel("$d$")
    plt.xticks(range(1,5))
    plt.ylabel("error(%)")
    plt.legend()
    #plt.savefig("degree_%s.png")
    plt.show()
    
    plt.plot(x_axis, nb_sv, 'k', label = "number of support vectors")
    plt.title("The number of support vectors over different d")
    plt.xlabel("$d$")
    plt.xticks(range(1,5))
    plt.legend()
    #plt.savefig("degree_%s.png")
    plt.show()
