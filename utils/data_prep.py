# for reading dataset

import scipy.io
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import tensorflow as tf

def get_list_all_training_frames(list_of_mat):
	
	pose3_ = []
	pose2_ = []
	files_ = []
	
	for ind, mFile in enumerate(list_of_mat):
		mat = scipy.io.loadmat(mFile)
		pose2_.append(mat['poses2'])
		pose3_.append(mat['poses3'])
		files_.append(mat['imgs'])
		ratio = 100*(float(ind)/float(len(list_of_mat)))
		if ratio%10 == 0:
			print('Successfully loaded -> ', ratio, '%')

	pose3 = np.concatenate(pose3_,axis=0)
	pose2 = np.concatenate(pose2_, axis=0)
	files = np.concatenate(files_, axis=0)

	return files, pose2, pose3


def get_list_all_testing_frames(list_of_mat):
	pose3_ = []
	pose2_ = []
	files_ = []
	gt_ = []
	for ind, mFile in enumerate(list_of_mat):
		mat = scipy.io.loadmat(mFile)
		pose2_.append(mat['poses2'])
		pose3_.append(mat['poses3'])
		files_.append(mat['imgs'])
		gt_.append(mat['p3GT'])
		ratio = 100 * (float(ind) / float(len(list_of_mat)))
		if ratio % 10 == 0:
			print('Successfully loaded -> ', ratio, '%')
	
	pose3 = np.concatenate(pose3_, axis=0)
	pose2 = np.concatenate(pose2_, axis=0)
	files = np.concatenate(files_, axis=0)
	gt = np.concatenate(gt_, axis=0)
	return files, pose2, pose3, gt


def get_batch(imgFiles, pose2, pose3, FLAG):
	data = []
	for name in imgFiles:
		im = misc.imread(name[:])
		data.append(im)
	return np.array(data), pose2, pose3

def crop_data_top_down(images, pose2, pose3):
	num_data_points = np.shape(images)[0]
	images_ = []
	pose2_ = []
	pose3_ = []
	for ii in xrange(num_data_points):
		im = images[ii]
		imSize = min(np.shape(im)[1], np.shape(im)[0])
		p2 = pose2[ii]# + Cam_C
		p3 = pose3[ii]
		#p3[:,0:2] = p3[:,0:2] + Cam_C
		p3[:,2] = p3[:,2]  # Means Res = 2cm per index for 64
		min_ = np.min(p2,axis=0)
		max_ = np.max(p2,axis=0)
		hW = np.max(max_ - min_)
		midP = np.mean(p2,axis=0)
		
		verSkw  = np.random.uniform(0.3, 0.7)
		horizSkw = np.random.uniform(0.35, 0.5)
		incSiz = np.random.uniform(60,80)
		#hW /= 2
		hW += incSiz
		skw = [verSkw,horizSkw]
		min_ = midP - skw * np.array(hW)
		


		
		hW = hW.astype(np.int)
		min_[1] = max(min_[1], 0)
		min_[0] = max(min_[0], 0)
		max_[1] = min((min_[1] + hW), imSize)
		max_[0] = min((min_[0] + hW), imSize)
		
		# Debugging Stuff
		#implot = plt.imshow(im)
		#plt.scatter(x=p2[:, 0], y=p2[:, 1], c='r')
		#plt.scatter(midP[0], midP[1], c='b')
		#plt.scatter(min_[0], min_[1], c='b')
		#plt.scatter(max_[0], max_[1], c='g')
		
		#plt.show()
		min_ = min_.astype(np.int)
		max_ = max_.astype(np.int)
		im_ = im[min_[1]:max_[1],min_[0]:max_[0]]
		p2 -= min_
		p3[:,:2] -= min_
		
		images_.append(im_)
		pose2_.append(p2)
		pose3_.append(p3)
	
	return images_,pose2_,pose3_

def data_vis(image, pose2, pose3, Cam_C, ind):
	im = image[ind]
	p2 = pose2[ind]
	p3 = pose3[ind]
	implot = plt.imshow(im)
	plt.scatter(x=p2[:,0],y=p2[:,1],c='r')
	plt.scatter(x=p3[:,0],y=p3[:,1],c='b')
	plt.show()
	
def gaussian(x, mu, sig, max_prob):
  return max_prob * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def plot_3d(image, threshold=0.5):
	# Position the scan upright,
	# so the head of the patient would be at the top facing the camera
	p = image.transpose(2, 1, 0)
	p = p[:, :, ::-1]
	
	verts, faces = measure.marching_cubes(p, threshold)
	
	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(111, projection='3d')
	
	# Fancy indexing: `verts[faces]` to generate a collection of triangles
	mesh = Poly3DCollection(verts[faces], alpha=0.1)
	face_color = [0.5, 0.5, 1]
	mesh.set_facecolor(face_color)
	ax.add_collection3d(mesh)
	
	ax.set_xlim(0, p.shape[0])
	ax.set_ylim(0, p.shape[1])
	ax.set_zlim(0, p.shape[2])
	
	plt.show()


def volumize_gt(image_b, pose2_b, pose3_b, resize_factor, im_resize_factor, \
                                                sigma, mul_factor, max_prob, FLAG):
	num_of_data = FLAG.batch_size
	batch_data = np.empty((0,14,resize_factor,resize_factor,resize_factor))
	pose2 = []
	pose3 = []
	image = []
	for ii in xrange(num_of_data):
		#print (ii, im_resize_factor, np.shape(image_b[ii]))
		im_ = misc.imresize(image_b[ii],(im_resize_factor,im_resize_factor))
		size_scale_ = np.array(np.shape(image_b[ii])[:2], dtype=np.float) / \
		              np.array(resize_factor, dtype=np.float)
		p2_ = pose2_b[ii] / size_scale_
		p3_ = pose3_b[ii]
		p3_[:,0:2] = p3_[:,0:2] / size_scale_
		p3_[:,2] = p3_[:,2] / np.mean(size_scale_)
		p3_[:, 2] *= mul_factor
		p3_[:,2] += resize_factor/2
		
		
		vec_x = np.empty((1,resize_factor))
		vec_y = np.empty((1,resize_factor))
		vec_z = np.empty((1,resize_factor))
		volume_b = np.empty((0,resize_factor,resize_factor,resize_factor))
		#vol_joint_ = np.zeros((64,64,64))
		for jj in xrange(14):
			for kk in xrange(resize_factor):
				vec_x[0,kk] = gaussian(kk, p3_[jj,0], sigma, max_prob)
				vec_y[0,kk] = gaussian(kk, p3_[jj, 1], sigma, max_prob)
				vec_z[0,kk] = gaussian(kk, p3_[jj, 2], sigma, max_prob)
			bub = np.expand_dims( vec_y.transpose().dot(vec_x),axis = 0)
			vol_joint = np.tensordot(bub, vec_z.transpose(), axes=([0],[1]))
			vol_joint = np.expand_dims(vol_joint,axis=0)
			volume_b = np.concatenate((volume_b,vol_joint),axis=0)
			#vol_joint_ += vol_joint
		#plot_3d(vol_joint_)
		volume_b = np.expand_dims(volume_b,axis=0)
		batch_data = np.concatenate((batch_data,volume_b),axis=0)
		pose2.append(p2_)
		pose3.append(p3_)
		image.append(im_)
		
	return batch_data,image, pose2, pose3


def get_vector_gt(image_b, pose2_b, pose3_b, FLAG):
	num_of_data = FLAG.batch_size
	vec_x = np.empty((FLAG.batch_size,FLAG.num_joints,FLAG.volume_res))
	vec_y = np.empty((FLAG.batch_size, FLAG.num_joints, FLAG.volume_res))
	vec_z = np.empty((FLAG.batch_size, FLAG.num_joints, FLAG.volume_res))
	pose2 = []
	pose3 = []
	image = np.empty((FLAG.batch_size, FLAG.image_res, FLAG.image_res, 3))
	
	for ii in xrange(num_of_data):
		# print (ii, im_resize_factor, np.shape(image_b[ii]))
		im_ = misc.imresize(image_b[ii], (FLAG.image_res, FLAG.image_res))
		size_scale_ = np.array(np.shape(image_b[ii])[:2], dtype=np.float) / \
		              np.array(FLAG.volume_res, dtype=np.float)
		p2_ = pose2_b[ii] / size_scale_
		p3_ = pose3_b[ii]
		p3_[:, 0:2] = p3_[:, 0:2] / size_scale_
		p3_[:, 2] = p3_[:, 2] / np.mean(size_scale_)
		p3_[:, 2] *= FLAG.mul_factor
		p3_[:, 2] += FLAG.volume_res / 2
		
		for jj in xrange(14):
			for kk in xrange(FLAG.volume_res):
				vec_x[ii, jj, kk] = gaussian(kk, p3_[jj, 0], FLAG.sigma, FLAG.joint_prob_max)
				vec_y[ii, jj, kk] = gaussian(kk, p3_[jj, 1], FLAG.sigma, FLAG.joint_prob_max)
				vec_z[ii, jj, kk] = gaussian(kk, p3_[jj, 2], FLAG.sigma, FLAG.joint_prob_max)
		
		pose2.append(p2_)
		pose3.append(p3_)
		image[ii,:,:,:] = im_
	
	return image, pose2, pose3, vec_x, vec_y, vec_z

def volumize_vec_gpu(tensor_x, tensor_y, tensor_z, FLAG):
	"""
	
	:param tensor_x: Probability distribution of GroundTruth along x axis
	:param tensor_y: Probability distribution of GroundTruth along y axis
	:param tensor_z: Probability distribution of GroundTruth along z axis
	:param FLAG: Parameters
	:return: Volumized representation for all joints in form of
	Batch - X - Y - Z - Joints
	
	"""
	list_b = []
	for ii in xrange(FLAG.batch_size):
		list_j = []
		for jj in xrange(FLAG.num_joints):
			vol = tf.matmul(tf.transpose(tensor_y[ii, jj:jj + 1]),
			                tensor_x[ii, jj:jj + 1])
			vol = tf.reshape(vol,[1,FLAG.volume_res, FLAG.volume_res])
			vol = tf.tensordot( vol, tf.transpose(tensor_z[ii, jj:jj + 1]),  axes=[[
				0],[1]])
			vol = tf.expand_dims(vol,3)
			list_j.append(vol)
		list_b.append(tf.concat(list_j,3))
	return tf.stack(list_b,0)

def prepare_output(batch_data,steps = [1, 2, 4, 64]):
	out_res = np.shape(batch_data)[0]
	batch_size = np.shape(batch_data)[1]
	output = np.empty((0,batch_size,14,out_res,out_res))
	for ii in steps:
		slice_ind = out_res / ii
		slice_start = 0
		for slice_end in range(slice_ind - 1, out_res, slice_ind):
			out_i = np.empty((0, 14, out_res, out_res))
			for data in xrange(batch_size):
				out_ = np.empty((0, out_res, out_res))
				vol_joint_ = np.zeros((64, 64, 64))
				for j in xrange(14):
					data_j = batch_data[:,data,:,:,j]
					slice_ = np.sum(data_j[slice_start:slice_end+1,:,:],axis=0)
					
					slice_ = np.expand_dims(slice_,axis=0)
					out_ = np.concatenate((out_,slice_),axis=0)
					
				out_ = np.expand_dims(out_,axis=0)
				out_i = np.concatenate((out_i,out_),axis=0)
			out_i = np.expand_dims(out_i,axis=0)
			output = np.concatenate((output,out_i),axis=0)
			slice_start = slice_end + 1
			
	return np.array(output)

def prepare_output_gpu(batch_data,steps, FLAG):
	"""input dims are
		# Batch - X - Y - Z - Joints
		We Want #Batch - X- Y- Z_*Joints"""
	list_b = []
	for b in xrange(FLAG.batch_size):
		list_fna = []
		for ss in steps:
			slice_ind = FLAG.volume_res / ss
			slice_start = 0
			for slice_end in xrange(slice_ind-1,FLAG.volume_res,slice_ind):
				list_j = []
				for jj in xrange(FLAG.num_joints):
					app = tf.expand_dims(tf.reduce_sum(batch_data[b,:,:,slice_start:slice_end+1,jj],2),2)
					list_j.append(tf.expand_dims(app,3))
				list_fna.append(tf.concat(list_j,3))
				slice_start = slice_end + 1
		list_b.append(tf.concat(list_fna,2))
	out_ =  tf.stack(list_b,0)
	return tf.reshape(out_,[FLAG.batch_size,FLAG.volume_res,FLAG.volume_res,FLAG.num_joints*sum(steps)])
