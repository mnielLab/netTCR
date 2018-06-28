#!/usr/bin/env python

"""
predict peptide binding to TCRs (author Vanessa Jurtz 2018)
"""

# Vanessa Jurtz 2018

from __future__ import print_function
import argparse
import sys
import os
import time
import random
import re
import csv
from operator import itemgetter

import math
import numpy as np
import tensorflow as tf

import data_io_tf
import NN_tf

config = tf.ConfigProto(intra_op_parallelism_threads=2, inter_op_parallelism_threads=2, allow_soft_placement=True)

################################################################################
#   SUBROUTINES
################################################################################

def iterate_minibatches(pep, tcr, targets, batchsize):
    assert pep.shape[0] == tcr.shape[0] == targets.shape[0]
    # shuffle:
    indices = np.arange(len(pep))
    np.random.shuffle(indices)
    for start_idx in range(0, len(pep), batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        yield pep[excerpt],tcr[excerpt],targets[excerpt]


################################################################################
#	PARSE COMMANDLINE OPTIONS
################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('-peptides', '--peptides',  help="specify peptides (comma separated)")
parser.add_argument('-infile', '--infile',  help="input file")
parser.add_argument('-outfile', '--outfile',  help="output file")
parser.add_argument('-param_dir', '--param_dir',  help="output directory",default="parameters/")
parser.add_argument('-batch_size', '--batch_size',  help="Mini batch size, default = 20", default=20)
parser.add_argument('-blosum', '--blosum', help="file with BLOSUM matrix", default="parameters/BLOSUM50")
parser.add_argument('-gpu', '--gpu', action="store_true", help="Use GPU, default = False", default=False)
args = parser.parse_args()


# get input peptides:
peplist=None
if args.peptides != None:
    print("# Specified peptides: " + args.peptides )
    peplist = args.peptides.upper().split(",")


# get input data (peptide + TCR or just TCR list):
if args.infile != None:
    print("# Input: " + args.infile )
    inputfile = args.infile
else:
    sys.stderr.write("Please specify input!\n")
    sys.exit(2)

# get output data:
if args.outfile != None:
    print("# Output file: " + args.outfile )
    outfile = open(args.outfile, "w")
else:
    outfile = open("NetTCR_predictions.txt", "w")

# get parameter directory:
try:
    print("# Parameter directory: " + args.param_dir )
    param_dir = args.param_dir
except:
    sys.stderr.write("Please specify parameter directory!\n")
    sys.exit(2)

# get mini-batch size:
try:
    BATCH_SIZE=int(args.batch_size)
    print("# batch size: " + str(BATCH_SIZE))
except:
    sys.stderr.write("Problem with mini batch size specification (option -batch_size)!\n")
    sys.exit(2)

# file with blosum matrix:
try:
    blosumfile=args.blosum
    print("# Blosum matrix file: " + str(blosumfile))
except:
    sys.stderr.write("Blosum encoding requires blosum matrix file!\n")
    sys.exit(2)

# Set to GPU:
if args.gpu==True:
    device_name = "/gpu:0"
else:
    device_name = "/cpu:0"

# SET SEED:
np.random.seed(1)
tf.set_random_seed(1)


################################################################################
#   LOAD AND PARTITION DATA
################################################################################
print("# Loading data...")

# read data:
if peplist==None:
    X_pep,X_tcr=data_io_tf.read_pTCR(inputfile)
else:
    X_pep,X_tcr=data_io_tf.read_pTCR_peplist(inputfile, peplist)

pep_aa = np.array(X_pep)
tcr_aa = np.array(X_tcr)

# encode data (list of numpy nd-arrays):
blosum = data_io_tf.read_blosum_MN(blosumfile)
X_pep = data_io_tf.enc_list_bl(X_pep, blosum)
X_tcr = data_io_tf.enc_list_bl_start_stop(X_tcr, blosum)

################################################################################
#   PREDICT NETWORKS
################################################################################
print("# predicting data...")

# prepare output file:
outfile.write("# NetTCR 1.0 predictions (Vanessa Jurtz et al. 2018)\n")
outfile.write("peptide\ttcr\tprediction\n")

all_pred=[]
N_PART=range(0,5)
s=1

# predict all partitions:
for t in N_PART:
    for v in N_PART:
        if t != v:
            # set up network:
            MODEL = np.load(param_dir + "params.t." + str(t) +  ".v." + str(v) + ".s." + str(s) + ".npz")['arr_1']
            hyper_params = np.load(param_dir + "params.t." + str(t) +  ".v." + str(v) + ".s." + str(s) + ".npz")['arr_2']
            N_FEATURES=int(hyper_params[0])
            N_HID=int(hyper_params[1])
            N_FILTERS=int(hyper_params[2])
            with tf.device(device_name):
                if MODEL=="CNN_opt2":
                    predictions,l_in_pep,l_in_tcr,drop_rate = NN_tf.build_CNN(n_features=N_FEATURES,n_hid=N_HID,n_filters=N_FILTERS)
                else:
                    sys.stderr.write("Error in model specification!\n")
                    sys.exit(2)
            # start tf session:
            sess = tf.Session(config=config)
            # variable initialization:
            sess.run(tf.global_variables_initializer())

            # set parameters:
            best_params = np.load(param_dir + "params.t." + str(t) +  ".v." + str(v) + ".s." + str(s) + ".npz")['arr_0']
            params=tf.trainable_variables()

            for i in range(0,len(params)):
                sess.run(params[i].assign(best_params[i]))

            # predict model:
            all_pred.append( sess.run(predictions, feed_dict={l_in_pep: X_pep, l_in_tcr: X_tcr, drop_rate: 0.0}) )

            # close session:
            sess.close()
            tf.reset_default_graph()

# calculate mean predictions of all models:
all_pred=np.array(all_pred)
pred = np.mean(all_pred, axis=0)


for i in range(0,pep_aa.shape[0]):
    outfile.write(str(pep_aa[i]) + "\t" + str(tcr_aa[i]) + "\t"  + str(pred[i][0]) + "\n")
outfile.write("# Done\n")
outfile.close()
