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
import matplotlib.pyplot as plt

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
                                                           pose3_b,64, 256,2)
print(np.shape(batch_data))
batch_data = np.swapaxes(batch_data,0,1) #swap Batch - Joint
batch_data = np.swapaxes(batch_data,0,2) #swap Joint - Depth
batch_data = np.swapaxes(batch_data,2,4) #swap Joint - X
batch_data = np.swapaxes(batch_data,0,2) #swap Depth - X#
# X - Batch - Depth - Y - Joint
print(np.shape(batch_data))

batch_output = utils.data_prep.prepare_output(batch_data)
batch_output = np.swapaxes(batch_output,2,4)
batch_output = np.swapaxes(batch_output,2,3)
fig = plt.figure()
a=fig.add_subplot(1,2,1)
plt.imshow(np.sum(batch_output[0,0,:,:,:],axis=2))
a=fig.add_subplot(1,2,2)
plt.imshow(image[0])
plt.show()

utils.data_prep.plot_3d(np.sum(batch_output[7:71,0,:,:,:],axis=3))

print (np.shape(batch_output))

#Wouldnt it be a great idea to test the fucked up code that i just wrote
#Lets do that by first ploting 3D model
print ( np.shape(np.sum(batch_output[7:71,0,:,:,:], axis=3)))
#utils.data_prep.plot_3d( np.sum(batch_output[7:71,0,:,:,:],
#                                               axis=1))

#The Great Parameter of Steps
#Choose it wisely
steps = [1, 2, 4, 64]

with tf.Graph().as_default():
	#Testing with only one GPU as of now
	DEVICE = '/gpu:0'
	#Assign the DEvice
	with tf.device(DEVICE):
		#first Build Model and define all the place holders
		# This can go to another file under name model
		# These parameters can be read from a external config file
		print ("start build model...")
		_x = tf.placeholder(tf.float32, [None, 256, 256, 3])
		y = tf.placeholder(tf.float32, [np.sum(np.array(steps)), None, 64,
                                    64, 14])
		# Calling external stacked_hourglass Function
		#ToDo : Change the hourglass implementation and make it more coustomizable
		output = hg.stacked_hourglass(np.sum(np.array(steps)),'stacked_hourglass')(_x)
		#Defining Loss with root mean square error
		loss = tf.reduce_mean(tf.square(output - y))
		#Defining optimizer over loss
		rmsprop = tf.train.RMSPropOptimizer(2.5e-4)
		print ("build finished, There she stands, tall and strong...")
    
	train_step = tf.Variable(0, name='global_step', trainable=False)
	with tf.device(DEVICE):
		train_rmsprop = rmsprop.minimize(loss, train_step)
  
	# Initializing all the variable in TF
	# Noting that the function depends on the version of the TF
	init = tf.global_variables_initializer()
  
	#Now standard TF session and training loops resides inside this fu*ker
	with tf.Session() as sess:
		with tf.device(DEVICE):
		# All the variable initialiezed in MoFoking RunTime
		#Confusing the world gets when yoda asks initializer operator before
			sess.run(init)
			print ("Let the Training Begin...")
			xarr = np.random.rand(100, 6, 256, 256, 3)
			yarr = np.random.rand(100, 4, 6, 64, 64, 14)
			_time = time.clock()
			with tf.device(DEVICE):
				sess.run(train_rmsprop, feed_dict={_x:image, y:batch_output})
			print ("test:", time.clock() - _time)