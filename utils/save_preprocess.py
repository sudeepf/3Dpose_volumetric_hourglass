from __future__ import print_function


from os import listdir
import os
from os.path import isfile, join
import numpy as np
from scipy import misc
import h5py

import tensorflow as tf

import sys
import time
import matplotlib.pyplot as plt

#Get all the custom helper util headers
import utils.data_prep
import utils.train_utils
import utils.eval_utils
import utils.get_flags
from six.moves import cPickle as pickle
#Read up and set up all the flag variables
FLAG = utils.get_flags.get_flags()



def main(_):
	if not FLAG.dataset_dir:
		raise ValueError('You must supply the dataset directory with --dataset_dir')
	
	_time = time.clock()
	DataHolder = utils.train_utils.DataHolder(FLAG)
	print('Time To lead the mats', time.clock() - _time)
	
	print('data loaded... phhhh')
	
	for step in range(DataHolder.train_data_size//FLAG.batch_size):
		image, pose2, pose3, imgFiles_, vec_x, vec_y, vec_z  = \
			DataHolder.get_preprocessed()
		print(str(imgFiles_)[3:])
		name = str(imgFiles_[1:]).split('.j')[0] + '.pickle'
		with open(name, 'wb') as pf:
			pickle.dump((image, pose2, pose3, vec_x, vec_y, vec_z), pf,
			            pickle.HIGHEST_PROTOCOL)
		print('saved',(imgFiles_[1:]))
		
if __name__ == '__main__':
	tf.app.run()

		