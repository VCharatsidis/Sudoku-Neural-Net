# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 16:07:23 2018

@author: vcharatsidis
"""
import tensorflow as tf
import numpy as np
from Sudoku import SolvedSudoku

hardestSudoku = [[8, 1, 2, 7, 5, 3, 6, 4, 9],
                 [9, 4, 3, 6, 8, 2, 1, 7, 5],
                 [6, 7, 5, 4, 9, 1, 2, 8, 3],
                 [1, 5, 4, 2, 3, 7, 8, 9, 6],
                 [3, 6, 9, 8, 4, 5, 7, 2, 1],
                 [2, 8, 7, 1, 6, 9, 5 ,3, 4],
                 [5, 2, 1, 9, 7, 4, 3, 6, 8],
                 [4, 3, 8, 5, 2, 6, 9, 1, 7],
                 [7, 9, 6, 3, 1, 8, 4, 5, 2]]

hardestSudoku_fixed = [[True, False, False, False, False, False, False, False, False],
                       [False, False, True, True, False, False, False, False, False],
                       [False, True, False, False, True, False, True, False, False],
                       [False, True, False, False, False, True, False, False, False],
                       [False, False, False, False, True, True, True, False, False],
                       [False, False, False, True, False, False, False, True, False],
                       [False, False, True, False, False, False, False, True, True],
                       [False, False, True, True, False, False, False, True, False],
                       [False, True, False, False, False, False, True, False, False]]

def one_hot(array):
    targets = np.array(np.asarray(array).reshape(-1))
    one_hot = np.eye(10)[targets]
    return one_hot

reducer = SolvedSudoku(hardestSudoku, hardestSudoku_fixed)

nodes1 = 810

x = tf.placeholder("float32", [1, 810], name = "reducedBoards")
y = tf.placeholder("float32", [1, 810], name = "solutions")

def nnmodel(data):
    hl1 = {'weights' : tf.Variable(tf.random_normal([810, nodes1])),
           'biases' : tf.Variable(tf.random_normal([nodes1]))}
    
    output_layer = {'weights' : tf.Variable(tf.random_normal([nodes1, 810])),
           'biases' : tf.Variable(tf.random_normal([810]))}
    
    lay1 = tf.matmul(data, hl1['weights']) + hl1['biases']
    lay1 = tf.nn.relu(lay1)
    
    output = tf.matmul(lay1, output_layer['weights']) + output_layer['biases']
    
    print("hi")

    return output

def prepare_data(x):
    ohx = one_hot(x)
    x_reshaped = np.reshape(ohx, 810)
    x_final = np.asarray([x_reshaped])
    
    return x_final

def train_nn(x):
    prediction = nnmodel(x)
    cost = tf.losses.mean_squared_error(labels = y, predictions = prediction) # 0.4-0.6
    #cost = tf.losses.absolute_difference(labels = y, predictions = prediction) 0.4-0.8
    #cost = tf.losses.log_loss(labels = y, predictions = prediction) nan
   
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        for i in range(30000):
            xs = reducer.board_to_row(reducer.board_reduction(1))
            x_prepared = prepare_data(xs)
            
            ys = reducer.board_to_row(reducer.solution)
            y_prepared = prepare_data(ys)
        
            _, c = sess.run([optimizer, cost], feed_dict = {x: x_prepared, y:y_prepared})
            if (i % 3000) == 0:
                print("cost "+str(c))
        
train_nn(x) 

      
        
        
        
