import tensorflow as tf
import utils.get_flags

FLAG = utils.get_flags.get_flags()


def gaussian_gpu(x, mu, sig, max_prob):
	n_ = tf.constant(-1,tf.float32)
	pow_ = tf.constant(2.0,tf.float32)
	return max_prob * tf.exp(n_*tf.pow(x-mu,pow_) / (pow_*tf.pow(sig,pow_)) )


def main(_):
	with tf.Graph().as_default():
		batch_data = tf.placeholder(tf.float32,
		                            [FLAG.batch_size, FLAG.volume_res,
		                             FLAG.volume_res,
		                             FLAG.volume_res, FLAG.num_joints])
		
		tensor_x = tf.placeholder(tf.float32,[FLAG.batch_size, FLAG.num_joints,
		                                      FLAG.volume_res])
		tensor_y = tf.placeholder(tf.float32, [FLAG.batch_size, FLAG.num_joints,
		                                       FLAG.volume_res])
		tensor_z = tf.placeholder(tf.float32, [FLAG.batch_size, FLAG.num_joints,
		                                       FLAG.volume_res])
		p3_ = tf.placeholder(tf.float32,[FLAG.batch_size, FLAG.num_joints, 3])
	
		sig_t = tf.constant(FLAG.sigma, tf.float32)
		max_prob_t = tf.constant(FLAG.joint_prob_max, tf.float32)
		for ii in xrange(FLAG.batch_size):
			
			
			for jj in xrange(FLAG.num_joints):
				
				
				vol = tf.matmul(tensor_y[ii,jj:jj+1],tf.transpose(tensor_x[ii,jj:jj+1]))
				vol = tf.matmul(vol,tensor_z[ii,jj:jj+1])
			
		with tf.Session() as sess:
				
			summary_writer = tf.summary.FileWriter('./',
		                                        graph_def=sess.graph)

if __name__ == '__main__':
	tf.app.run()