from __future__ import print_function

#import torch
import tensorflow as tf
from os import listdir
import os
from os.path import isfile, join
from scipy import misc


import sys
import numpy as np
import time
import hourglass_TF.src.stacked_hourglass as hg
import matplotlib.pyplot as plt

#Get all the custom helper util headers
import utils.data_prep
import utils.add_summary
import data_prep
import utils.get_flags

class DataHolder():
	def __init__(self, FLAG):
		self.FLAG = FLAG
		# get all the subjects
		train_subjects = FLAG.data_split_string_train.split('-')
		test_subjects = FLAG.data_split_string_test.split('-')
		
		# Train Folder and Test Folder
		print('Train Dataset: -> ', train_subjects)
		print('Test Dataset: -> ', test_subjects)
		
		# list all the matfiles
		# make one for test and one for train
		self.mFiles_train = []
		for ind, folder in enumerate(train_subjects):
			mat_folder_path = join(join(FLAG.dataset_dir, folder), 'mats')
			mFiles_ = [join(join(join(FLAG.dataset_dir, folder), 'mats'), f) for f in
			           listdir(mat_folder_path) if f.split('.')[-1] == 'mat']
			self.mFiles_train += mFiles_
		print('Total Train Actions x Subjects x Acts -> ', len(self.mFiles_train))
		
		self.mFiles_test = []
		for ind, folder in enumerate(test_subjects):
			mat_folder_path = join(join(FLAG.dataset_dir, folder), 'mats')
			mFiles_ = [join(join(join(FLAG.dataset_dir, folder), 'mats'), f) for f in
			           listdir(mat_folder_path) if f.split('.')[-1] == 'mat']
			self.mFiles_test += mFiles_
		print('Total Test Actions x Subjects x Acts -> ', len(self.mFiles_test))
		
		self.read_mat_files()
		
		self.train_data_size = np.shape(self.imgFiles)[0]
		self.test_data_size = np.shape(self.imgFiles_test)[0]
		
		print('Total Training Data Frames are ', self.train_data_size)
		print('Total Testing Data Frames are ', self.test_data_size)
		
		#initializing training and testing iterations
		self.train_iter = 0
		self.test_iter = 0
		
		#Getting Suffled Mask
		self.mask_train = np.random.permutation(self.train_data_size)
		self.mask_test = np.random.permutation(self.test_data_size)
		
	def read_mat_files(self):
		self.imgFiles, self.pose2, self.pose3 = \
			data_prep.get_list_all_training_frames(self.mFiles_train)
		
		self.imgFiles_test, self.pose2_test, self.pose3_test = \
			data_prep.get_list_all_training_frames(self.mFiles_test)
		
	def get_dict(self, train, imgFiles, pose2, pose3):
		"""Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
		steps = map(int, self.FLAG.structure_string.split('-'))
		total_dim = np.sum(np.array(steps))
		
		if train or not (train):
			image_b, pose2_b, pose3_b = utils.data_prep.get_batch(imgFiles,
			                                                      pose2, pose3,
			                                                      self.FLAG)
			
			image_b, pose2_b, pose3_b = utils.data_prep.crop_data_top_down(image_b,
			                                                               pose2_b,
			                                                               pose3_b)
			
			
			batch_data, image, pose2, pose3 = utils.data_prep.volumize_gt(image_b,
			                                                              pose2_b,
			                                                              pose3_b,
			                                                              self.FLAG.volume_res,
			                                                              self.FLAG.image_res,
			                                                              self.FLAG.sigma,
			                                                              self.FLAG.mul_factor,
			                                                              self.FLAG.joint_prob_max)
			
			# Batch - Joints - X - Y - Z
			batch_data = np.swapaxes(batch_data, 1, 4)  # swap Z - Joint
			# Batch - Z - X - Y - Joints
			batch_data = np.swapaxes(batch_data, 0, 1)  # swap Joint - Depth
			# Z- Batch - X - Y - Joints
			
			batch_output = utils.data_prep.prepare_output(batch_data, steps)
			# 3D - Batch - Joints - X - Y
			batch_output = np.rollaxis(batch_output, 0, 5)
			# Batch - J - X - Y - 3D
			batch_output = np.rollaxis(batch_output, 1, 5)
			# Batch - X - Y - 3D - Joints
			
			# from string model struct to list model struct
			
			
			batch_output = np.reshape(batch_output,
			                          (self.FLAG.batch_size, self.FLAG.volume_res,
			                           self.FLAG.volume_res,
			                           total_dim * self.FLAG.num_joints))
		
		else:
			print('nothing')
		# xs, ys = mnist.test.images, mnist.test.labels
		# k = 1.0
		return image, batch_output, pose3

	def get_next_train_batch(self):
		"""
		:return: This Function basically gives out next batch data everytime its
		been called, as it has all the information it needs, I totally needed
		this function now my life becomes much simpler
		"""
		
		offset = min((self.train_iter * self.FLAG.batch_size) , \
		         (self.train_data_size - self.FLAG.batch_size))
		mask_ = self.mask_train[offset:(offset + self.FLAG.batch_size)]
		
		fd = self.get_dict(True, self.imgFiles[mask_], self.pose2[mask_],
		              self.pose3[mask_])
		
		self.train_iter += 1
		
		if self.train_iter * self.FLAG.batch_size > self.train_data_size:
			print('<<<<<<<<<<<<<<<<<:Epoch:>>>>>>>>>>>>>>>>> ')
			self.train_iter = 0
			self.mask_train = np.random.permutation(self.train_data_size)
			
		return fd[0], fd[1], fd[2]
	
	def get_preprocessed(self):
		"""
		returns preprocessed data for saving
		:return:
		"""
		
		
		offset = min((self.train_iter * 1), \
		             (self.train_data_size - 1))
		
		imgFiles_ =  self.imgFiles[offset:(offset + 1)]
		pose2_ = self.pose2[offset:(offset + 1)]
		pose3_ = self.pose3[offset:(offset + 1)]
		
		image_b, pose2_b, pose3_b = utils.data_prep.get_batch(imgFiles_,
		                                                      pose2_, pose3_,
		                                                      self.FLAG)
		
		image_b, pose2_b, pose3_b = utils.data_prep.crop_data_top_down(image_b,
		                                                               pose2_b,
		                                                               pose3_b)
		
		num_of_data = np.shape(image_b)[0]
		
		pose2 = []
		pose3 = []
		image = []
		
		vec_x = np.empty((self.FLAG.num_joints, self.FLAG.volume_res))
		vec_y = np.empty((self.FLAG.num_joints, self.FLAG.volume_res))
		vec_z = np.empty((self.FLAG.num_joints, self.FLAG.volume_res))
		
		for ii in xrange(num_of_data):
			# print (ii, im_resize_factor, np.shape(image_b[ii]))
			im_ = misc.imresize(image_b[ii], (self.FLAG.image_res,
			                                  self.FLAG.image_res))
			size_scale_ = np.array(np.shape(image_b[ii])[:2], dtype=np.float) / \
			              np.array(self.FLAG.volume_res, dtype=np.float)
			p2_ = pose2_b[ii] / size_scale_
			p3_ = pose3_b[ii]
			p3_[:, 0:2] = p3_[:, 0:2] / size_scale_
			p3_[:, 2] = p3_[:, 2] / np.mean(size_scale_)
			p3_[:, 2] *= self.FLAG.mul_factor
			p3_[:, 2] += self.FLAG.volume_res / 2
			
			
			
			
			for jj in xrange(14):
				for kk in xrange(self.FLAG.volume_res):
					vec_x[jj, kk] = utils.data_prep.gaussian(kk, p3_[jj, 0],
					                                         self.FLAG.sigma,
					                                         self.FLAG.joint_prob_max)
					vec_y[jj, kk] = utils.data_prep.gaussian(kk, p3_[jj, 1], self.FLAG.sigma,
					                                         self.FLAG.joint_prob_max)
					vec_z[jj, kk] = utils.data_prep.gaussian(kk, p3_[jj, 2], self.FLAG.sigma,
					                                         self.FLAG.joint_prob_max)
					
			pose2.append(p2_)
			pose3.append(p3_)
			image.append(im_)
		
		self.train_iter += 1
		
		
		return image, pose2, pose3, imgFiles_, vec_x, vec_y, vec_z