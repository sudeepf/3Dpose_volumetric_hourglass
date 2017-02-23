from __future__ import print_function

#import torch
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

#Get all the custom helper util headers
import utils.data_prep
import utils.add_summary
import utils.train_utils
import utils.eval_utils
import utils.get_flags
import include.hg_graph_builder


#Read up and set up all the flag variables
FLAG = utils.get_flags.get_flags()


def main(_):
  if not FLAG.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')
  
  DataHolder = utils.train_utils.DataHolder(FLAG)

  print('data loaded... phhhh')
	

  with tf.Graph().as_default():
		
		builder = include.hg_graph_builder.HGgraphBuilder(FLAG)
		
		print ("build finished, There it stands, tall and strong...")
		
		
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
	
		saver = tf.train.Saver()
	
		with tf.Session(config=config) as sess:
			merged = tf.summary.merge_all()
	
			# All the variable initialiezed in MoFoking RunTime
			# Confusing the world gets when yoda asks initializer operator before
	
	
			train_writer, test_writer=utils.add_summary.get_summary_writer(FLAG, sess)
			
			ckpt = tf.train.get_checkpoint_state(FLAG.load_ckpt_path)
			
			if ckpt and ckpt.model_checkpoint_path:
				if os.path.isabs(ckpt.model_checkpoint_path):
					# Restores from checkpoint with absolute path.
					saver.restore(sess, ckpt.model_checkpoint_path)
				else:
					# Restores from checkpoint with relative path.
					saver.restore(sess, os.path.join(FLAG.load_ckpt_path,
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
	
			
	
			print ("Let the Training Begin...")
	
			# Train the model, and also write summaries.
			# Every 10th step, measure test-set accuracy, and write test summaries
			# All other steps, run train_step on training data, & add training summaries
			structure = map(int, FLAG.structure_string.split('-'))
	
			for step in range(DataHolder.train_data_size):
					
					if step % 100 == 99:  # Record execution stats
							save_path = saver.save(sess,
							                       FLAG.eval_dir+"/tmp/model" +
							                       FLAG.structure_string+".ckpt")
							print('Adding Model data for ', step, 'at ', save_path)
					
					if step % 1000 == 999:  # Record execution stats
							save_path = saver.save(sess, FLAG.model_path + '/model_%05d_' %
							                       step + FLAG.structure_string  +'.ckpt')
							print('Adding Model data for ', step, 'at ', save_path)
					
					fd = DataHolder.get_next_train_batch()
					
					if step % 10 == 1:
						gt_ = fd[2]
						summary, loss_, out_test = sess.run([merged, builder.loss,
						                                     builder.output],
						                             feed_dict={builder._x: fd[0],
						                                        builder.y: fd[1]})
						err = utils.eval_utils.compute_precision(out_test, gt_,
						                                          structure,
						                                          FLAG.mul_factor,
						                                          FLAG.num_joints)
						
						train_writer.add_summary(summary, step)
						print ("Current Accuracy is" + str(np.sum(err)))
						continue
					
					
					loss_, _ = sess.run([builder.loss, builder.train_rmsprop],
					                             feed_dict={builder._x: fd[0],
					                                        builder.y:  fd[1]})
					
					print("Grinding... Loss = " + str(loss_))
			
			train_writer.close()
			test_writer.close()
		  
if __name__ == '__main__':
	tf.app.run()
