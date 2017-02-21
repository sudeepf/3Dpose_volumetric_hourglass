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
import utils.train_utils

import sys
import numpy as np
import time
import hourglass_TF.src.stacked_hourglass as hg
import matplotlib.pyplot as plt

#Path to the Dataset or the subject folders

data_path = './Dataset/'
#model_path = './Reference/pose-hg-train-master/src/models'
model_path ='./models/'

#Camera Params
Cam_C = [512.53, 515.49]
Cam_F = [1143, 1146]

#get all the subjects
onlyFolders = [f for f in listdir(data_path) if isfile(join(data_path, f))!=1]
onlyFolders.sort()

#Train Folder and Test Folder
onlyFolders_train = onlyFolders[:]
print('Train Dataset: -> ',onlyFolders_train)
onlyFolders_test = onlyFolders[:]
print('Test Dataset: -> ',onlyFolders_test)

#list all the matfiles
#make one for test and one for train
mFiles_train = []
for ind, folder in enumerate(onlyFolders_train):
	mat_folder_path = join(join(data_path,folder),'mats')
	mFiles_ = [join(join(join(data_path,folder),'mats'), f) for f in listdir(
		mat_folder_path) if f.split('.')[-1] == 'mat']
	mFiles_train += mFiles_
print('Total Train Actions x Subjects x Acts -> ', len(mFiles_train))

mFiles_test = []
for ind, folder in enumerate(onlyFolders_train):
	mat_folder_path = join(join(data_path,folder),'mats')
	mFiles_ = [join(join(join(data_path,folder),'mats'), f) for f in listdir(
		mat_folder_path) if f.split('.')[-1] == 'mat']
	mFiles_test += mFiles_
print('Total Train Actions x Subjects x Acts -> ', len(mFiles_test))

# Parameters
batch_size = 2
volume_res = 64
num_joints = 14
mul_factor = 500
sigma = 1
image_res = 256
#The Great Parameter of Steps
#Choose it wisely
steps = [64]
#steps = [1,1]
total_dim = np.sum(np.array(steps))
summery_path = './tensor_record/'
# Read all the mat files and merge the training data

imgFiles, pose2, pose3, scale = utils.data_prep.get_list_all_training_frames(
	mFiles_train)

imgFiles_test, pose2_test, pose3_test, scale_test = \
	utils.data_prep.get_list_all_training_frames(
	mFiles_test)


data_size = np.shape(imgFiles)[0]

def feed_dict(train, imgFiles, pose2, pose3):
	"""Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
	if train or not(train):
		image_b, pose2_b, pose3_b = utils.data_prep.get_batch(imgFiles,
		                                                      pose2, pose3)
		
		image_b, pose2_b, pose3_b = utils.data_prep.crop_data_top_down(image_b,
		                                                               pose2_b,
		                                                               pose3_b)
		
		# Visualization of data to check if all went right
		#utils.data_prep.data_vis(image_b, pose2_b, pose3_b, Cam_C, 0)
		
		
		batch_data, image, pose2, pose3 = utils.data_prep.volumize_gt(image_b,
		                                                              pose2_b,
		                                                              pose3_b,
		                                                              volume_res,
		                                                              image_res,
		                                                              sigma,
		                                                              mul_factor)
		# print(np.shape(batch_data))
		# Batch - Joints - X - Y - Z
		batch_data = np.swapaxes(batch_data, 1, 4)  # swap Z - Joint
		# Batch - Z - X - Y - Joints
		batch_data = np.swapaxes(batch_data, 0, 1)  # swap Joint - Depth
		# Z- Batch - X - Y - Joints
		
		# print(np.shape(batch_data))
		
		batch_output = utils.data_prep.prepare_output(batch_data,steps)
		#print(np.shape(batch_output))
		# 3D - Batch - Joints - X - Y
		batch_output = np.rollaxis(batch_output, 0, 5)
		# Batch - J - X - Y - 3D
		#print(np.shape(batch_output))
		# 3D - X - Batch - J - Y
		# Batch - Joints - X - Y - 3D
		# print(np.shape(batch_output))
		batch_output = np.rollaxis(batch_output, 1, 5)
		# Batch - X - Y - 3D - Joints
		# print(np.shape(batch_output))
		# print(np.shape(batch_output[0,:,:,0,:]))
		#print (np.shape(batch_output))
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
	return image, batch_output, pose3

#TESTINGGGGG TO BE DELETED

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
	gt = tf.placeholder(tf.float32, [None, num_joints,
                                  3])
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
	
	#Accuracy for love of God!
	
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
		
		ckpt = tf.train.get_checkpoint_state(model_path)
		if ckpt and ckpt.model_checkpoint_path:
			if os.path.isabs(ckpt.model_checkpoint_path):
				# Restores from checkpoint with absolute path.
				saver.restore(sess, ckpt.model_checkpoint_path)
			else:
				# Restores from checkpoint with relative path.
				saver.restore(sess, os.path.join(model_path,
				                                 ckpt.model_checkpoint_path))
			
			# Assuming model_checkpoint_path looks something like:
			#   /my-favorite-path/imagenet_train/model.ckpt-0,
			# extract global_step from it.
			global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
			print('Succesfully loaded model from %s at step=%s.' %
			      (ckpt.model_checkpoint_path, global_step))
		else:
				print('No checkpoint file found')
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
				if step % 50 == 499:  # Record summaries and test-set accuracy
						fD = feed_dict(True, imgFiles[mask_], pose2[mask_], pose3[mask_])
						gt_ = pose3[mask_]
						
						summary, loss_, out_test = sess.run([merged, loss, output],
						                               feed_dict={_x: fD[0], y:fD[1]})
						err = utils.train_utils.compute_precision(out_test, gt_, steps,
						                                          mul_factor, num_joints)
						summary_ = tf.Summary()
						summary_.ParseFromString(sess.run(summary))
						utils.add_summary.add_all_joints(err,summary_)
						
						test_writer.add_summary(summary, step)
						print('Loss at step %s: %s' % (step, loss_))
				
				if step % 5000 == 4999:  # Record summaries and test-set accuracy
					fD = feed_dict(True, imgFiles, pose2, pose3)
					gt_ = pose3
					
					summary, loss_, out_test = sess.run([merged, loss, output],
					                                    feed_dict={_x: fD[0], y: fD[1]})
					
					err = utils.train_utils.compute_precision(out_test, gt_, steps,
					                                          scale[mask_],
					                                          mul_factor, num_joints)
					test_writer.add_summary(summary, step)
					print('Loss at step %s: %s' % (step, loss_))
				
				
		
				if step % 100 == 99:  # Record execution stats
						save_path = saver.save(sess, summery_path+"/tmp/model.ckpt")
						print('Adding Model data for ', step, 'at ', save_path)
				
				if step % 1000 == 999:  # Record execution stats
						save_path = saver.save(sess, model_path + '/model_%05d' % step +'.ckpt')
						print('Adding Model data for ', step, 'at ', save_path)
				
				fD = feed_dict(True, imgFiles[mask_], pose2[mask_], pose3[mask_])
				summary, loss_, _ = sess.run([merged, loss, train_rmsprop],
																 feed_dict={_x: fD[0], y: fD[1]})
				
				if step % 10 == 1:
					gt_ = pose3[mask_]
					summary, loss_, out_test = sess.run([merged, loss, output],
					                             feed_dict={_x: fD[0], y: fD[1]})
					err = utils.train_utils.compute_precision(out_test, gt_, steps,
					                                          mul_factor, num_joints)
					summary_ = tf.Summary()
					summary_.ParseFromString(sess.run(summary))
					utils.add_summary.add_all_joints(err,summary_)
					train_writer.add_summary(summary_, step)
					print ("Current Accuracy is" + str(np.sum(err)))
				print("Grinding... Loss = " + str(loss_))
		
		train_writer.close()
		test_writer.close()
