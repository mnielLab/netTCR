#!/usr/bin/env python

"""
Functions for specifiying neural network architectures.
"""

from __future__ import print_function
import argparse
import sys
import os
import time

import math
import numpy as np
import tensorflow as tf


def build_CNN(n_features, n_hid, n_filters):

    # input:
    l_in_pep = tf.placeholder(tf.float32, shape=[None,None,n_features])
    l_in_tcr = tf.placeholder(tf.float32, shape=[None,None,n_features])
    drop_rate = tf.placeholder(tf.float32, shape=())

    # convolutional layers on peptide:
    l_conv_pep_1 = tf.layers.conv1d(inputs=l_in_pep, filters=n_filters, kernel_size=1, kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same", activation=tf.nn.sigmoid)
    l_conv_pep_3 = tf.layers.conv1d(inputs=l_in_pep, filters=n_filters, kernel_size=3, kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same", activation=tf.nn.sigmoid)
    l_conv_pep_5 = tf.layers.conv1d(inputs=l_in_pep, filters=n_filters, kernel_size=5, kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same", activation=tf.nn.sigmoid)
    l_conv_pep_7 = tf.layers.conv1d(inputs=l_in_pep, filters=n_filters, kernel_size=7, kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same", activation=tf.nn.sigmoid)
    l_conv_pep_9 = tf.layers.conv1d(inputs=l_in_pep, filters=n_filters, kernel_size=9, kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same", activation=tf.nn.sigmoid)

    # convolutional layers on TCR:
    l_conv_tcr_1 = tf.layers.conv1d(inputs=l_in_tcr, filters=n_filters, kernel_size=1, kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same", activation=tf.nn.sigmoid)
    l_conv_tcr_3 = tf.layers.conv1d(inputs=l_in_tcr, filters=n_filters, kernel_size=3, kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same", activation=tf.nn.sigmoid)
    l_conv_tcr_5 = tf.layers.conv1d(inputs=l_in_tcr, filters=n_filters, kernel_size=5, kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same", activation=tf.nn.sigmoid)
    l_conv_tcr_7 = tf.layers.conv1d(inputs=l_in_tcr, filters=n_filters, kernel_size=7, kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same", activation=tf.nn.sigmoid)
    l_conv_tcr_9 = tf.layers.conv1d(inputs=l_in_tcr, filters=n_filters, kernel_size=9, kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same", activation=tf.nn.sigmoid)

    # second convolutional layer:
    l_conc_1 = tf.concat([l_conv_pep_1, l_conv_tcr_1], axis=1)
    l_conc_3 = tf.concat([l_conv_pep_3, l_conv_tcr_3], axis=1)
    l_conc_5 = tf.concat([l_conv_pep_5, l_conv_tcr_5], axis=1)
    l_conc_7 = tf.concat([l_conv_pep_7, l_conv_tcr_7], axis=1)
    l_conc_9 = tf.concat([l_conv_pep_9, l_conv_tcr_9], axis=1)

    l_conv_2_1 = tf.layers.conv1d(inputs=l_conc_1, filters=n_filters, kernel_size=1, kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same", activation=tf.nn.sigmoid)
    l_conv_2_3 = tf.layers.conv1d(inputs=l_conc_3, filters=n_filters, kernel_size=1, kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same", activation=tf.nn.sigmoid)
    l_conv_2_5 = tf.layers.conv1d(inputs=l_conc_5, filters=n_filters, kernel_size=1, kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same", activation=tf.nn.sigmoid)
    l_conv_2_7 = tf.layers.conv1d(inputs=l_conc_7, filters=n_filters, kernel_size=1, kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same", activation=tf.nn.sigmoid)
    l_conv_2_9 = tf.layers.conv1d(inputs=l_conc_9, filters=n_filters, kernel_size=1, kernel_initializer=tf.contrib.layers.xavier_initializer(), padding="same", activation=tf.nn.sigmoid)

    # max pooling:
    l_pool_max_1 = tf.reduce_max(l_conv_2_1,axis=1)
    l_pool_max_3 = tf.reduce_max(l_conv_2_3,axis=1)
    l_pool_max_5 = tf.reduce_max(l_conv_2_5,axis=1)
    l_pool_max_7 = tf.reduce_max(l_conv_2_7,axis=1)
    l_pool_max_9 = tf.reduce_max(l_conv_2_9,axis=1)

    # concatenate:
    l_conc = tf.concat([l_pool_max_1, l_pool_max_3, l_pool_max_5, l_pool_max_7, l_pool_max_9], axis=1)

    # dense hidden layer:
    l_dense = tf.layers.dense(inputs=l_conc, units=n_hid, activation=tf.nn.sigmoid)

    # dropout:
    l_dense_drop = tf.layers.dropout(inputs=l_dense, rate=drop_rate, noise_shape=None, seed=None, training=True)

    # output layer:
    out = tf.layers.dense(inputs=l_dense_drop, units=1, activation=tf.nn.sigmoid)

    return out,l_in_pep,l_in_tcr,drop_rate
