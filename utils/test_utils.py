from __future__ import print_function

# import torch
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
from mpl_toolkits.mplot3d import Axes3D

# Get all the custom helper util headers
import utils.data_prep
import utils.add_summary
import data_prep
import utils.get_flags


class TestDataHolder():
    def __init__(self, FLAG):
        # type: (object) -> object
        self.FLAG = FLAG
        # get all the subjects
        test_subjects = FLAG.data_split_string_test.split('-')
        
        # Train Folder and Test Folder
        print('Test Dataset: -> ', test_subjects)
        
        # list all the matfiles
        # make one for test and one for train
        
        self.mFiles_test = []
        for ind, folder in enumerate(test_subjects):
            mat_folder_path = join(join(FLAG.dataset_dir, folder), 'mats')
            mFiles_ = [join(join(join(FLAG.dataset_dir, folder), 'mats'), f) for
                       f in
                       listdir(mat_folder_path) if f.split('.')[-1] == 'mat']
            self.mFiles_test += mFiles_
        print('Total Test Actions x Subjects x Acts -> ', len(self.mFiles_test))
        
        self.read_mat_files()
        
        self.test_data_size = np.shape(self.imgFiles_test)[0]
        
        print('Total Testing Data Frames are ', self.test_data_size)
        
        # initializing training and testing iterations
        self.train_iter = 0
        
        # Getting Suffled Mask
        self.mask_test = np.random.permutation(self.test_data_size)
    
    def read_mat_files(self):
        self.imgFiles_test, self.pose2_test, self.pose3_test, self.gt = \
            data_prep.get_list_all_testing_frames(self.mFiles_test)
    
    def get_test_set(self, train, imgFiles, pose2, pose3):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        
        image_b, pose2_b, pose3_b = utils.data_prep.get_batch(imgFiles,
                                                              pose2, pose3,
                                                              self.FLAG)
        
        image_b, pose2_b, pose3_b = utils.data_prep.crop_data_top_down(image_b,
                                                                       pose2_b,
                                                                       pose3_b)
        
        image, pose2, pose3, vec_x, vec_y, vec_z = \
            utils.data_prep.get_vector_gt(image_b, pose2_b, pose3_b, self.FLAG)
        
        return image, vec_x, vec_y, vec_z, pose3
    
    def get_next_train_batch(self):
        """
		:return: This Function basically gives out next batch data everytime its
		been called, as it has all the information it needs, I totally needed
		this function now my life becomes much simpler
		"""
        
        offset = min((self.train_iter * self.FLAG.batch_size), \
                     (self.test_data_size - self.FLAG.batch_size))
        mask_ = [offset]#:(offset + self.FLAG.batch_size)]
        
        fd = self.get_test_set(True, self.imgFiles_test[mask_],
                               self.pose2_test[mask_],
                               self.pose3_test[mask_])
        
        self.train_iter += 1
        return fd[0], fd[1], fd[2], fd[3], fd[4], self.gt[mask_]


def visualize_stickman(cords, image, ind):
    fig = plt.figure()

    image_ = fig.add_subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    # Plotting all the limbs individually

    ax = fig.add_subplot(1,2,2, projection='3d')
    ax.plot([cords[0,0],cords[1,0]], [cords[0,1],cords[1,1]],  [cords[0,2],
                                                                cords[1,2]])
    ax.plot([cords[0, 0], cords[1, 0]], [cords[0, 1], cords[1, 1]],
            [cords[0, 2],
             cords[1, 2]],'b')
    ax.plot([cords[1, 0], cords[2, 0]], [cords[1, 1], cords[2, 1]],
            [cords[1, 2],
             cords[2, 2]],'g')
    ax.plot([cords[2, 0], cords[3, 0]], [cords[2, 1], cords[3, 1]],
            [cords[2, 2],
             cords[3, 2]],'r')
    ax.plot([cords[3, 0], cords[4, 0]], [cords[3, 1], cords[4, 1]],
            [cords[3, 2],
             cords[4, 2]],'m')
    ax.plot([cords[1, 0], cords[5, 0]], [cords[1, 1], cords[5, 1]],
            [cords[1, 2],
             cords[5, 2]],'g')
    ax.plot([cords[5, 0], cords[6, 0]], [cords[5, 1], cords[6, 1]],
            [cords[5, 2],
             cords[6, 2]],'r')
    ax.plot([cords[6, 0], cords[7, 0]], [cords[6, 1], cords[7, 1]],
            [cords[6, 2],
             cords[7, 2]],'m')
    ax.plot([cords[2, 0], cords[8, 0]], [cords[2, 1], cords[8, 1]],
            [cords[2, 2],
             cords[8, 2]],'c')
    ax.plot([cords[5, 0], cords[11, 0]], [cords[5, 1], cords[11, 1]],
            [cords[5, 2],
             cords[11, 2]],'c')
    ax.plot([cords[8, 0], cords[11, 0]], [cords[8, 1], cords[11, 1]],
            [cords[8, 2],
             cords[11, 2]],'c')
    ax.plot([cords[8, 0], cords[9, 0]], [cords[8, 1], cords[9, 1]],
            [cords[8, 2],
             cords[9, 2]],'k')
    ax.plot([cords[9, 0], cords[10, 0]], [cords[9, 1], cords[10, 1]],
            [cords[9, 2],
             cords[10, 2]],'r')
    ax.plot([cords[11, 0], cords[12, 0]], [cords[11, 1], cords[12, 1]],
            [cords[11, 2],
             cords[12, 2]],'b')
    ax.plot([cords[12, 0], cords[13, 0]], [cords[12, 1], cords[13, 1]],
            [cords[12, 2],
             cords[13, 2]],'r')
    
    ax.set_xlim3d(0, 64)
    ax.set_ylim3d(0, 64)
    ax.set_zlim3d(0, 64)
    ax.view_init(-50,-50)
    
    plt.show()
    plt.savefig('./vids/foo_'+str(ind)+'.png', bbox_inches='tight')