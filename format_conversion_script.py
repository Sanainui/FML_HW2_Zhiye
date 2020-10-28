# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 16:51:30 2020

@author: xzye0
"""
#label 0: rings <= 9; label 1: ring > 9

def sex_to_int(s):
    if s == 'M':
        return 1
    elif s == 'F':
        return 2
    elif s == 'I':
        return 3

file_in = open('abalone.data', 'r') 
file_out = open('abalone_formated.csv', 'w')

Lines = file_in.readlines()

for line in Lines:
    sex, length, diameter, height, whole_weight, shucked_weight, \
        viscera_weight, shell_weight, rings = line[:-1].split(',')

    o_str = '{} 1:{} 2:{} 3:{} 4:{} 5:{} 6:{} 7:{} 8:{}\n'.format\
        (int(int(rings) <= 9), sex_to_int(sex), length, diameter, height, whole_weight, \
         shucked_weight, viscera_weight, shell_weight)
    file_out.write(o_str)

file_in.close()
file_out.close()