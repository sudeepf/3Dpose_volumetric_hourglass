import tensorflow as tf
import numpy as np

def variable_summaries(var, name):
	"""Attach a lot of summaries to a Tensor  (for TensorBoard visualization)."""

	with tf.name_scope('summaries_' + name):
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean', mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
			tf.summary.scalar('stddev', stddev)
		tf.summary.scalar('max', tf.reduce_max(var))
		tf.summary.scalar('min', tf.reduce_min(var))
		tf.summary.histogram('histogram', var)
		
def image_summaries(img, name):
	with tf.name_scope('summaries_image_' + name):
		image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
		tf.summary.image('input_'+name, image_shaped_input, 10)

def vox2img(vox, steps):
	start_z = np.sum(steps[0:len(steps)-1])
	size = steps[-1]
	img = tf.reduce_sum(vox,[0,0,0,start_z],[1,64,64,size],2)
	
def add_all_joints(prec, summary, name='precision'):
	summary_ = tf.Summary()
	summary_.ParseFromString(sess.run(summary))
	for j, err in enumerate(prec):
		summary_.value.add(name+'joint_%02d' % j , err)
	#adding total error
	summary_.value.add(name+'total_error',np.sum(prec))