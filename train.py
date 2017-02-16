from __future__ import print_function

#import torch
import tensorflow as tf
from os import listdir
import os
from os.path import isfile, join
import numpy as np
from scipy import misc
import utils.data_prep
import sys
import numpy as np
import time
import hourglass_TF.src.stacked_hourglass as hg

#Path to the Dataset or the subject folders

data_path = './Dataset/'
model_path = './Reference/pose-hg-train-master/src/models'
#Camera Params
Cam_C = [512.53, 515.49]
Cam_F = [1143, 1146]

#get all the subjects
onlyFolders = [f for f in listdir(data_path) if isfile(join(data_path, f))!=1]
onlyFolders.sort()

#list all the matfiles
mFiles = []
for ind, folder in enumerate(onlyFolders):
	mat_folder_path = join(join(data_path,folder),'mats')
	mFiles_ = [join(join(join(data_path,folder),'mats'), f) for f in listdir(
		mat_folder_path) if f.split('.')[-1] == 'mat']
	mFiles += mFiles_
	
	
# Read all the mat files and merge the training data

imgFiles, pose2, pose3 = utils.data_prep.get_list_all_training_frames(mFiles)


image_b, pose2_b, pose3_b = utils.data_prep.get_batch(imgFiles,
                                                      pose2, pose3, 8)


image_b, pose2_b, pose3_b = utils.data_prep.crop_data_top_down(image_b,
                                                                pose2_b,
                                                               pose3_b, Cam_C)

#Visualization of data to check if all went right
#utils.data_prep.data_vis(image_b, pose2_b, pose3_b, Cam_C, 100)


batch_data,image, pose2, pose3 = utils.data_prep.volumize_gt(image_b,pose2_b,
                                                           pose3_b,64, 128,2)

batch_output = utils.data_prep.prepare_output(batch_data)

print (np.shape(batch_output))


with tf.Graph().as_default():
  DEVICE = '/gpu:0'
  with tf.device(DEVICE):
      print ("start build model...")
      _x = tf.placeholder(tf.float32, [None, 256, 256, 3])
      y = tf.placeholder(tf.float32, [4, None, 64, 64, 16])
      output = hg.stacked_hourglass(4,'stacked_hourglass')(_x)
      loss = tf.reduce_mean(tf.square(output - y))
      rmsprop = tf.train.RMSPropOptimizer(2.5e-4)
      print ("build finished...")
  train_step = tf.Variable(0, name='global_step', trainable=False)
  with tf.device(DEVICE):
      train_rmsprop = rmsprop.minimize(loss, train_step)
  init = tf.global_variables_initializer()
  with tf.Session() as sess:
      with tf.device(DEVICE):
          sess.run(init)
      print ("test...")
      xarr = np.random.rand(100, 6, 256, 256, 3)
      yarr = np.random.rand(100, 4, 6, 64, 64, 16)
      _time = time.clock()
      with tf.device(DEVICE):
          for u in range(0, 100):
              sess.run(train_rmsprop, feed_dict={_x:xarr[u], y:yarr[u]})
      print ("test:", time.clock() - _time)