from __future__ import print_function

# import torch
import tensorflow as tf
from os import listdir
import os
from os.path import isfile, join
import numpy as np
from scipy import misc

import sys
import numpy as np
import time
import hourglass_TF.src.stacked_hourglass as hg
import matplotlib.pyplot as plt

# Get all the custom helper util headers
import utils.data_prep
import utils.add_summary
import utils.test_utils
import utils.eval_utils
import utils.get_flags
import include.hg_graph_builder

# Read up and set up all the flag variables
FLAG = utils.get_flags.get_flags()


def main(_):
	if not FLAG.dataset_dir:
		raise ValueError('You must supply the dataset directory with --dataset_dir')
	if not FLAG.dataset_dir:
		raise ValueError('You must supply the model_path with --load_ckpt_path')
	DataHolder = utils.test_utils.TestDataHolder(FLAG)
	
	print('data loaded... phhhh')
	
	with tf.Graph().as_default():
		
		# builder = include.hg_graph_builder.HGgraphBuilder(FLAG)
		
		builder = include.hg_graph_builder.HGgraphBuilder_MultiGPU(FLAG)
		print("build finished, There it stands, tall and strong...")
		
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		
		saver = tf.train.Saver()
		
		with tf.Session(config=config) as sess:
			
			# All the variable initialiezed in MoFoking RunTime
			# Confusing the world gets when yoda asks initializer operator before
			
			
			print(FLAG.load_ckpt_path)
		
			saver.restore(sess, FLAG.load_ckpt_path)
			print('model Initialized...')
			
			print("Let the Testing Begin...")
			
			# Train the model, and also write summaries.
			# Every 10th step, measure test-set accuracy, and write test summaries
			# All other steps, run train_step on training data, & add training summaries
			yo = []
			for step in range(100):
				
				_x = []
				gt = []
				for i in map(int, FLAG.gpu_string.split('-')):
					fd = DataHolder.get_next_train_batch()
					_x.append(fd[0])
					gt.append(fd[5])
				feed_dict_x = {i: d for i, d in zip(builder._x, _x)}
					
				time_ = time.clock()
				output_ = sess.run([builder.output], feed_dict_x)
				print("Time to feed and run the network", time.clock() - time_)
				steps = map(int, FLAG.structure_string.split('-'))
				ypy = 0
				for idh in xrange(len(map(int, FLAG.gpu_string.split('-')))):
					ypy += utils.eval_utils.compute_precision(output_[idh][0], gt[idh],
					                                         steps, FLAG.mul_factor, 14)
				ypy /= len(map(int, FLAG.gpu_string.split('-')))
				print("Mean Error",np.sum(ypy)/14)
				yo.append(ypy)
				
			print("Total error",np.sum(np.sum(np.stack(
				yo)))/DataHolder.test_data_size)
			print ( "Mean Error", ((np.sum(np.sum(np.stack(
				yo)))/DataHolder.test_data_size)/14))
			

if __name__ == '__main__':
	tf.app.run()
