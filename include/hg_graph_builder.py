import hourglass_TF.src.stacked_hourglass as hg
import tensorflow as tf
import numpy as np
import utils.add_summary

class HGgraphBuilder():
	def __init__(self, FLAG):
		# first Build Model and define all the place holders
		# This can go to another file under name model
		# These parameters can be read from a external config file
		print ("start build model...")
		
		#from string model struct to list model struct
		steps = map(int, FLAG.structure_string.split('-'))
		total_dim = np.sum(np.array(steps))
		
		self._x = tf.placeholder(tf.float32, [None, FLAG.image_res, FLAG.image_res,
		                                 FLAG.image_c])
		self.y = tf.placeholder(tf.float32, [None, FLAG.volume_res,
		                                FLAG.volume_res, FLAG.num_joints *
		                                total_dim])
		
		# If I ever write a handle for accuracy computation in TF
		self.gt = tf.placeholder(tf.float32, [None, FLAG.num_joints,
		                                 3])
		
		self.output = hg.stacked_hourglass(steps, 'stacked_hourglass')(self._x)
		
		# Defining Loss with root mean square error
		self.loss = tf.reduce_mean(tf.square(self.output - self.y))
		
		self.optimizer = tf.train.RMSPropOptimizer(FLAG.learning_rate)
		
		self.train_step = tf.Variable(0, name='global_step', trainable=False)
		
		self.train_rmsprop = self.optimizer.minimize(self.loss, self.train_step)

		utils.add_summary.add_all(self._x,self.y,self.output,self.loss)