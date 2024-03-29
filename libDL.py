#-*- coding=utf-8 -*-
import math
import sys


"""
This code refers to the url:
https://github.com/palloc/1D-CNN-Library/blob/master/libDL.py
"""


# learning rate
epsilon = 0.5

# make weight
def MakeWeight(x, y):
    return [[1.0 for i in range(x)] for j in range(y)]


# read data
def Open_data(filename):
    input_file = open(filename)
    temp = input_file.read().split('\n')
    input_node = []
    for i in temp:
        if i != '':
            input_node.append(map(int, i.split(',')))
    return input_node


"""
---------------------------------
     Calculate Layer's node
---------------------------------
"""

# Logistic function (activation function)
def Logistic_Func(x):
    return 1.0 / float((1.0 + math.exp(-x)))

# Fully connected layer's Calc_func
def FullyConnected_Func(x, w):

    # error process (if can't mult matrix)
    if len(x) < len(w[0]):
        print("Multiplication Failed in FC layer.")
        return 0.0
    next_node = []

    # dot product
    for i in w:
        temp = 0.0
        for j in range(len(w)):
            temp += x[j] * i[j]
        next_node.append(temp)
    return next_node

# Convolution layer's Calc_func
def Conv_Func(x, w):

    #error process (if can't mult matrix)
    if len(x) < len(w):
        print("Multiplication Failed in Conv layer")
        return 0.0
    next_node = []
    counter = 0

    # Convolution
    for i in range(len(x) - len(w)):
        temp = 0.0

        # dot product without bias
        for j in range(counter, counter+len(w)-1):
            temp += x[j] * w[j-counter]
        
        next_node.append(temp)
        counter += 1
    return next_node

# Max pooling layer's Calc_func
def Max_Pool_Func(x, kernel_size):
    next_node = []
    counter = 0
    while counter < len(x):
        max_temp = 0.0

        # add max_node in the kernel to next_node
        for i in range(counter, counter + kernel_size - 1):
            if max_temp < x[i]:
                max_temp = x[i]
        next_node.append(max_temp)
        counter += kernel_size
    return next_node

# Pass fully conected layer
def Pass_FC(old_layer, new_layer, w):
    new_layer.bp_node = old_layer.node
    old_layer.node.append(1)
    new_layer.node = FullyConnected_Func(old_layer.node, w)
    new_layer.Do_Logistic()

# Pass convolution layer
def Pass_Conv(old_layer, new_layer, w):
    new_layer.bp_node = old_layer.node
    old_layer.node.append(1)
    new_layer.node = Conv_Func(old_layer.node, w)
    new_layer.Do_Logistic()

# Pass max pooling layer
def Pass_Max_Pool(old_layer, new_layer, kernel_size):
    new_layer.node = Max_Pool_Func(old_layer.node, kernel_size)

# Pass fully connected layer
def Pass_FC_Out(old_layer, new_layer, w):
    old_layer.node.append(1)
    new_layer.node = FullyConnected_Func(old_layer.node, w)
    new_layer.Do_Softmax()


"""
---------------------------------
  Function for back-propagation
---------------------------------
"""

# Differential logistic function
def Dif_Logistic_Func(x):
    new_node = []
    for i in x:
        new_node.append(Logistic_Func(i) * (1 - Logistic_Func(i)))
    return new_node

# Calculate cross entropy
def Cross_Entropy(x, d):
    result = []
    for i in range(len(x)):
        result.append(x[i] - d[i])
    return result




def main():
    x = [1, 2, 3]
    w = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    res = FullyConnected_Func(x, w)
    print(res)


if __name__ == '__main__':
    main()
