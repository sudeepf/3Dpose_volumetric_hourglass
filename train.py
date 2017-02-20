from __future__ import print_function

#import torch
import tensorflow as tf
from os import listdir
import os
from os.path import isfile, join
import numpy as np
from scipy import misc
import utils.data_prep
import utils.add_summary

import sys
import numpy as np
import time
import hourglass_TF.src.stacked_hourglass as hg
import matplotlib.pyplot as plt

#Path to the Dataset or the subject folders

data_path = '/home/capstone/datasets/Human3.6M/Subjects/'
#model_path = './Reference/pose-hg-train-master/src/models'
model_path ='./models/'

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
	if ind==1:
		break

# Parameters
batch_size = 8
volume_res = 64
num_joints = 14
#The Great Parameter of Steps
#Choose it wisely
steps = [1, 2, 4, 64]
#steps = [1,1]
total_dim = np.sum(np.array(steps))
summery_path = './tensor_record/'
# Read all the mat files and merge the training data

imgFiles, pose2, pose3 = utils.data_prep.get_list_all_training_frames(mFiles)

data_size = np.shape(imgFiles)[0]

def feed_dict(train, imgFiles, pose2, pose3, mask):
	"""Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
	if train or not(train):
		image_b, pose2_b, pose3_b = utils.data_prep.get_batch(imgFiles,
		                                                      pose2, pose3, batch_size, mask)
		
		image_b, pose2_b, pose3_b = utils.data_prep.crop_data_top_down(image_b,
		                                                               pose2_b,
		                                                               pose3_b,
		                                                               Cam_C)
		
		# Visualization of data to check if all went right
		# utils.data_prep.data_vis(image_b, pose2_b, pose3_b, Cam_C, 0)
		
		
		batch_data, image, pose2, pose3 = utils.data_prep.volumize_gt(image_b,
		                                                              pose2_b,
		                                                              pose3_b, 64,
		                                                              256, 2)
		# print(np.shape(batch_data))
		# Batch - Joints - X - Y - Z
		batch_data = np.swapaxes(batch_data, 1, 4)  # swap Z - Joint
		# Batch - Z - X - Y - Joints
		batch_data = np.swapaxes(batch_data, 0, 1)  # swap Joint - Depth
		# Z- Batch - X - Y - Joints
		
		# print(np.shape(batch_data))
		
		batch_output = utils.data_prep.prepare_output(batch_data,steps)
		# 3D - Batch - Joints - X - Y
		# print(np.shape(batch_output))
		batch_output = np.rollaxis(batch_output, 3, 1)
		# Batch - Joints - X - Y - 3D
		# print(np.shape(batch_output))
		batch_output = np.rollaxis(batch_output, 4, 2)
		# Batch - X - Y - 3D - Joints
		# print(np.shape(batch_output))
		# print(np.shape(batch_output[0,:,:,0,:]))
		
		#fig = plt.figure()
		#a=fig.add_subplot(1,2,1)
		#plt.imshow(np.sum(batch_output[0,:,:,0,:],axis=2))
		#a=fig.add_subplot(1,2,2)
		#plt.imshow(image[0])
		#plt.show()
		
		batch_output = np.reshape(batch_output, (batch_size, volume_res, volume_res,
		                                         total_dim * num_joints))
		
	else:
		print('nothing')
		#xs, ys = mnist.test.images, mnist.test.labels
		#k = 1.0
	return image, batch_output


# Build Model
with tf.Graph().as_default():
	#Testing with only one GPU as of now
	DEVICE = '/gpu:1'
	#Assign the DEvice
	#first Build Model and define all the place holders
	# This can go to another file under name model
	# These parameters can be read from a external config file
	print ("start build model...")
	_x = tf.placeholder(tf.float32, [None, 256, 256, 3])
	y = tf.placeholder(tf.float32, [None, 64,
                                  64, num_joints*total_dim])
	
	#get_out_img = utils.add_summary.vox2img(y, steps)
	
	# adding Label Summery
	utils.add_summary.variable_summaries(y,'label')
	#utils.add_summary.image_summaries(get_out_img,'label')
	
	# Calling external stacked_hourglass Function
	#ToDo : Change the hourglass implementation and make it more coustomizable
	output = hg.stacked_hourglass(steps,'stacked_hourglass')(_x)
	#Defining Loss with root mean square error
	loss = tf.reduce_mean(tf.square(output - y))
	tf.summary.scalar('loss', loss)
	
	#Defining optimizer over loss
	#get_out_img = utils.add_summary.vox2img(output, steps)
	
	# adding Label Summery
	utils.add_summary.variable_summaries(output,'pred')
	#utils.add_summary.image_summaries(get_out_img, 'pred')
	
	rmsprop = tf.train.RMSPropOptimizer(2.5e-3)
	#Printing Loss

	
	print ("build finished, There it stands, tall and strong...")
	
	train_step = tf.Variable(0, name='global_step', trainable=False)
	train_rmsprop = rmsprop.minimize(loss, train_step)

	# Initializing all the variable in TF
	# Noting that the function depends on the version of the TF
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	saver = tf.train.Saver()

	with tf.Session(config=config) as sess:
		merged = tf.summary.merge_all()

		# All the variable initialiezed in MoFoking RunTime
		#Confusing the world gets when yoda asks initializer operator before


		train_writer = tf.summary.FileWriter( summery_path + '/train', \
									 sess.graph)
		test_writer = tf.summary.FileWriter(summery_path + '/test')



		if os.path.isfile(summery_path+"/tmp/model.ckpt"):
				saver.restore(sess, summery_path+"/tmp/model.ckpt")
				print("Model restored.")
		else:
				tf.global_variables_initializer().run()
				print('model Initialized...')

		writer = tf.summary.FileWriter(summery_path,
																	 graph=tf.get_default_graph())

		print ("Let the Training Begin...")

		# Train the model, and also write summaries.
		# Every 10th step, measure test-set accuracy, and write test summaries
		# All other steps, run train_step on training data, & add training summaries


		mask = np.random.permutation(np.shape(imgFiles)[0])

		for step in range(data_size):
				offset = (step * batch_size) % (data_size - batch_size)
				mask_ = mask[offset:(offset + batch_size)]
				#mask_ = mask[0:batch_size]
				if step % 50 == 0:  # Record summaries and test-set accuracy
						fD = feed_dict(True, imgFiles, pose2, pose3, mask_)
						print ( np.shape(fD[1]))
						summary, loss_ = sess.run([merged, loss], feed_dict={_x: fD[0],
																																y:fD[1]})
						test_writer.add_summary(summary, step)
						print('Loss at step %s: %s' % (step, loss_))
				else:  # Record train set summaries, and train
						if step % 100 == 99:  # Record execution stats
								save_path = saver.save(sess, summery_path+"/tmp/model.ckpt")
								print('Adding Model data for ', step, 'at ', save_path)
						if step % 1000 == 999:  # Record execution stats
								save_path = saver.save(sess, model_path + '/model_%05d' % step +'.ckpt')
								print('Adding Model data for ', step, 'at ', save_path)
						fD = feed_dict(True, imgFiles, pose2, pose3, mask_)
						summary, loss_, _ = sess.run([merged, loss, train_rmsprop],
																		 feed_dict={_x: fD[0], y: fD[1]})
						train_writer.add_summary(summary, step)
						print("Grinding... Loss = " + str(loss_))
		train_writer.close()
		test_writer.close()
