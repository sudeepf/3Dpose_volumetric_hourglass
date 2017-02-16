
import tensorflow as tf

class stacked_hourglass():
	def __init__(self, nb_stack, name='stacked_hourglass'):
		self.nb_stack = nb_stack
		self.name = name
	
	def __call__(self, x):
		with tf.name_scope(self.name) as scope:
			padding = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], name='padding')
			with tf.name_scope("preprocessing") as sc:
				conv1 = self._conv(padding, 64, 7, 2, 'VALID', 'conv1')
				norm1 = tf.contrib.layers.batch_norm(conv1, 0.9, epsilon=1e-5,
				                                     activation_fn=tf.nn.relu, scope=sc)
				r1 = self._residual_block(norm1, 128, 'r1')
				pool = tf.contrib.layers.max_pool2d(r1, [2, 2], [2, 2], 'VALID',
				                                    scope=scope)
				r2 = self._residual_block(pool, 128, 'r2')
				r3 = self._residual_block(r2, 256, 'r3')
			hg = [None] * self.nb_stack
			ll = [None] * self.nb_stack
			ll_ = [None] * self.nb_stack
			out = [None] * self.nb_stack
			out_ = [None] * self.nb_stack
			sum_ = [None] * self.nb_stack
			with tf.name_scope('_hourglass_0_with_supervision') as sc:
				hg[0] = self._hourglass(r3, 4, 256, '_hourglass')
				ll[0] = self._conv_bn_relu(hg[0], 256, name='conv_1')
				ll_[0] = self._conv(ll[0], 256, 1, 1, 'VALID', 'll')
				out[0] = self._conv(ll[0], 16, 1, 1, 'VALID', 'out')
				out_[0] = self._conv(out[0], 256, 1, 1, 'VALID', 'out_')
				sum_[0] = tf.add_n([ll_[0], out_[0], r3])
			for i in range(1, self.nb_stack - 1):
				with tf.name_scope('_hourglass_' + str(i) + '_with_supervision') as sc:
					hg[i] = self._hourglass(sum_[i - 1], 4, 256, '_hourglass')
					ll[i] = self._conv_bn_relu(hg[i], 256, name='conv_1')
					ll_[i] = self._conv(ll[i], 256, 1, 1, 'VALID', 'll')
					out[i] = self._conv(ll[i], 16, 1, 1, 'VALID', 'out')
					out_[i] = self._conv(out[i], 256, 1, 1, 'VALID', 'out_')
					sum_[i] = tf.add_n([ll_[i], out_[i], sum_[i - 1]])
			with tf.name_scope(
						'_hourglass_' + str(self.nb_stack - 1) + '_with_supervision') as sc:
				hg[self.nb_stack - 1] = self._hourglass(sum_[self.nb_stack - 2], 4, 256,
				                                        '_hourglass')
				ll[self.nb_stack - 1] = self._conv_bn_relu(hg[self.nb_stack - 1], 256,
				                                           name='conv_1')
				out[self.nb_stack - 1] = self._conv(ll[self.nb_stack - 1], 16, 1, 1,
				                                    'VALID', 'out')
			return tf.stack(out)
	
	def _conv(self, inputs, nb_filter, kernel_size=1, strides=1, pad='VALID',
	          name='conv'):
		with tf.name_scope(name) as scope:
			kernel = tf.Variable(
				tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size, \
				                                                     kernel_size,
				                                                     inputs.get_shape().as_list()[
					                                                     3], nb_filter]),
				name='weights')
			conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], padding=pad,
			                    data_format='NHWC')
			return conv
	
	def _conv_bn_relu(self, inputs, nb_filter, kernel_size=1, strides=1,
	                  name=None):
		with tf.name_scope(name) as scope:
			kernel = tf.Variable(
				tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size, \
				                                                     kernel_size,
				                                                     inputs.get_shape().as_list()[
					                                                     3], nb_filter]),
				name='weights')
			conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1],
			                    padding='SAME', data_format='NHWC')
			norm = tf.contrib.layers.batch_norm(conv, 0.9, epsilon=1e-5,
			                                    activation_fn=tf.nn.relu, scope=scope)
			return norm
	
	def _conv_block(self, inputs, nb_filter_out, name='_conv_block'):
		with tf.name_scope(name) as scope:
			with tf.name_scope('norm_conv1') as sc:
				norm1 = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5,
				                                     activation_fn=tf.nn.relu, scope=sc)
				conv1 = self._conv(norm1, nb_filter_out / 2, 1, 1, 'SAME', name='conv1')
			with tf.name_scope('norm_conv2') as sc:
				norm2 = tf.contrib.layers.batch_norm(conv1, 0.9, epsilon=1e-5,
				                                     activation_fn=tf.nn.relu, scope=sc)
				conv2 = self._conv(norm2, nb_filter_out / 2, 3, 1, 'SAME', name='conv2')
			with tf.name_scope('norm_conv3') as sc:
				norm3 = tf.contrib.layers.batch_norm(conv2, 0.9, epsilon=1e-5,
				                                     activation_fn=tf.nn.relu, scope=sc)
				conv3 = self._conv(norm3, nb_filter_out, 1, 1, 'SAME', name='conv3')
			return conv3
	
	def _skip_layer(self, inputs, nb_filter_out, name='_skip_layer'):
		if inputs.get_shape()[3].__eq__(tf.Dimension(nb_filter_out)):
			return inputs
		else:
			with tf.name_scope(name) as scope:
				conv = self._conv(inputs, nb_filter_out, 1, 1, 'SAME', name='conv')
				return conv
	
	def _residual_block(self, inputs, nb_filter_out, name='_residual_block'):
		with tf.name_scope(name) as scope:
			_conv_block = self._conv_block(inputs, nb_filter_out)
			_skip_layer = self._skip_layer(inputs, nb_filter_out)
			return tf.add(_skip_layer, _conv_block)
	
	def _hourglass(self, inputs, n, nb_filter_res, name='_hourglass'):
		with tf.name_scope(name) as scope:
			# Upper branch
			up1 = self._residual_block(inputs, nb_filter_res, 'up1')
			# Lower branch
			pool = tf.contrib.layers.max_pool2d(inputs, [2, 2], [2, 2], 'VALID',
			                                    scope=scope)
			low1 = self._residual_block(pool, nb_filter_res, 'low1')
			if n > 1:
				low2 = self._hourglass(low1, n - 1, nb_filter_res, 'low2')
			else:
				low2 = self._residual_block(low1, nb_filter_res, 'low2')
			low3 = self._residual_block(low2, nb_filter_res, 'low3')
			low4 = tf.image.resize_nearest_neighbor(low3, tf.shape(low3)[1:3] * 2,
			                                        name='upsampling')
			if n < 4:
				return tf.add(up1, low4, name='merge')
			else:
				return self._residual_block(tf.add(up1, low4), nb_filter_res, 'low4')
