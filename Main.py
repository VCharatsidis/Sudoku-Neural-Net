# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 12:49:15 2018

@author: vcharatsidis
"""

from Sudoku import SolvedSudoku
import tensorflow as tf
import numpy as np

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

reducer = SolvedSudoku(hardestSudoku, hardestSudoku_fixed)

learning_rate = 0.01
training_iteration = 10000
batch_size = 1
display_step = 500

x = tf.placeholder("float", [None, 81], name = "reducedBoards")
y = tf.placeholder("float", name = "solutions")

W = tf.Variable(tf.zeros([81,81]))
b = tf.Variable(tf.zeros([81]))

with tf.name_scope("Wx_b") as scope:
    model = tf.nn.softmax(tf.matmul(x, W) + b)
    
with tf.name_scope("cost_function") as scope:
    cost_function = -tf.reduce_sum(y * tf.log(model))
    tf.summary.scalar("cost_function", cost_function)
    
with tf.name_scope("train") as scope:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

init = tf.initialize_all_variables()

merged_summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    
    for iteration in range(training_iteration):
        avg_cost = 0.
        
        xs = reducer.board_to_row(reducer.board_reduction(1))
        xs = np.array(xs).reshape(1, 81)
        
        ys = reducer.board_to_row(reducer.solution)
        ys = np.array(ys).reshape(1, 81)
        #batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        
        sess.run(optimizer, feed_dict = {x: xs, y: ys})
        
        avg_cost += sess.run(cost_function, feed_dict = {x: xs, y: ys}) / training_iteration
        
        summary_str = sess.run(merged_summary_op, feed_dict = {x: xs, y: ys})
       # summary_writer.add_summary(summary_str, iteration = total_batch +i)
            
        if iteration % display_step == 0:
            print("Iteration:", '%04d' % (iteration+1), "cost = ", "{:.9f}".format(avg_cost))
            
    print("Tuning completed!")
    