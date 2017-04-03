import cv2 as cv
import numpy as np
import scipy
import PIL.Image
import math
import caffe
import time
from config_reader import config_reader
import util
import copy
import matplotlib
import pylab as plt

import utils.data_prep
import utils.add_summary
import data_prep
import utils.get_flags

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
import matplotlib.animation as animation
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
        raise ValueError(
            'You must supply the dataset directory with --dataset_dir')
    if not FLAG.dataset_dir:
        raise ValueError('You must supply the model_path with --load_ckpt_path')
    DataHolder = utils.test_utils.TestDataHolder(FLAG)
    
    print('data loaded... phhhh')
    
    
    for step in range(DataHolder.train_data_size):
        fd = DataHolder.get_next_train_batch()
        param, model = config_reader()
        multiplier = [x * model['boxsize'] / oriImg.shape[0] for x in
                      param['scale_search']]
        if param['use_gpu']:
            caffe.set_mode_gpu()
            caffe.set_device(param['GPUdeviceNumber'])  # set to your device!
        else:
            caffe.set_mode_cpu()
        net = caffe.Net(model['deployFile'], model['caffemodel'], caffe.TEST)

        heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
        paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))
        

        for m in range(len(multiplier)):
            scale = multiplier[m]
            imageToTest = cv.resize(oriImg, (0, 0), fx=scale, fy=scale,
                                    interpolation=cv.INTER_CUBIC)
            imageToTest_padded, pad = util.padRightDownCorner(imageToTest,
                                                              model['stride'],
                                                              model['padValue'])
            print
            imageToTest_padded.shape
    
            axarr[m].imshow(imageToTest_padded[:, :, [2, 1, 0]])
            axarr[m].set_title('Input image: scale %d' % m)
    
            net.blobs['data'].reshape(*(
            1, 3, imageToTest_padded.shape[0], imageToTest_padded.shape[1]))
            # net.forward() # dry run
            net.blobs['data'].data[...] = np.transpose(
                np.float32(imageToTest_padded[:, :, :, np.newaxis]),
                (3, 2, 0, 1)) / 256 - 0.5;
            start_time = time.time()
            output_blobs = net.forward()
            print('At scale %d, The CNN took %.2f ms.' % (
            m, 1000 * (time.time() - start_time)))
    
            # extract outputs, resize, and remove padding
            heatmap = np.transpose(
                np.squeeze(net.blobs[output_blobs.keys()[1]].data),
                (1, 2, 0))  # output 1 is heatmaps
            heatmap = cv.resize(heatmap, (0, 0), fx=model['stride'],
                                fy=model['stride'],
                                interpolation=cv.INTER_CUBIC)
            heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2],
                      :imageToTest_padded.shape[1] - pad[3], :]
            heatmap = cv.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]),
                                interpolation=cv.INTER_CUBIC)
    
            paf = np.transpose(
                np.squeeze(net.blobs[output_blobs.keys()[0]].data),
                (1, 2, 0))  # output 0 is PAFs
            paf = cv.resize(paf, (0, 0), fx=model['stride'], fy=model['stride'],
                            interpolation=cv.INTER_CUBIC)
            paf = paf[:imageToTest_padded.shape[0] - pad[2],
                  :imageToTest_padded.shape[1] - pad[3], :]
            paf = cv.resize(paf, (oriImg.shape[1], oriImg.shape[0]),
                            interpolation=cv.INTER_CUBIC)
    
            # visualization
            heatmap_avg = heatmap_avg + heatmap / len(multiplier)
            paf_avg = paf_avg + paf / len(multiplier)
            
            plt.imshow(np.sum(heatmap_avg,axis=-1))
            plt.show()
            

if __name__ == '__main__':
    tf.app.run()
