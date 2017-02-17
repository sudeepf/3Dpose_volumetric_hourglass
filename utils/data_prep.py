# for reading dataset

import scipy.io
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def get_list_all_training_frames(list_of_mat):
	pose2 = np.empty((0,14,2))
	pose3 = np.empty((0,14,3))
	files = np.empty((0))
	for mFile in list_of_mat:
		print(mFile)
		mat = scipy.io.loadmat(mFile)
		pose2 = np.concatenate((pose2,mat['poses2']),axis=0)
		pose3 = np.concatenate((pose3, mat['poses3']), axis=0)
		files = np.concatenate((files, mat['imgs']), axis=0)
	return files, pose2, pose3

def shuffle_data(imgFiles, pose2, pose3):
	mask = np.random.permutation(np.shape(imgFiles)[0])
	imgFiles_ = imgFiles[mask]
	pose2_ = pose2[:num, :, :]
	pose3_ = pose3[:num, :, :]
	return imgFiles_, pose2_, pose3_

def get_batch(imgFiles, pose2, pose3, num):
	mask = np.random.permutation(np.shape(imgFiles)[0])[:num]
	imgFiles_ = imgFiles[mask]
	pose2_ = pose2[mask,:,:]
	pose3_ = pose3[mask,:,:]
	data = []
	for name in imgFiles_:
		im = misc.imread(name[1:])
		data.append(im)
	return np.array(data), pose2_, pose3_

def crop_data_top_down(images, pose2, pose3, Cam_C):
	num_data_points = np.shape(images)[0]
	images_ = []
	pose2_ = []
	pose3_ = []
	for ii in xrange(num_data_points):
		im = images[ii]
		p2 = pose2[ii]# + Cam_C
		p3 = pose3[ii]
		#p3[:,0:2] = p3[:,0:2] + Cam_C
		p3[:,2] = p3[:,2]  # Means Res = 2cm per index for 64
		min_ = np.min(p2,axis=0)
		max_ = np.max(p2,axis=0)
		hW = np.max(max_ - min_)
		midP = np.mean(p2,axis=0)
		horizSkw  = np.random.uniform(0.3, 0.7)
		verSkw = np.random.uniform(0.3, 0.7)
		incSiz = np.random.uniform(70,100)
		hW /= 2
		hW += incSiz
		skw = [verSkw,horizSkw]
		min_ = midP -  skw * np.array(hW*2)
		min_ = min_.astype(np.int)
		hW *= 2
		hW = hW.astype(np.int)
		im_ = im[min_[1]:(min_[1]+hW),min_[0]:(min_[0]+hW)]
		p2 -= min_
		p3[:,:2] -= min_
		#print(min_ , midP, skw , np.array(hW))
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
	
def gaussian(x, mu, sig):
  return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


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
                                                        sigma):
	num_of_data = np.shape(image_b)[0]
	batch_data = np.empty((0,14,resize_factor,resize_factor,resize_factor))
	pose2 = []
	pose3 = []
	image = []
	for ii in xrange(num_of_data):
		im_ = misc.imresize(image_b[ii],(im_resize_factor,im_resize_factor))
		size_scale_ = np.array(np.shape(image_b[ii])[:2], dtype=np.float) / \
		              np.array(resize_factor, dtype=np.float)
		p2_ = pose2_b[ii] / size_scale_
		p3_ = pose3_b[ii]
		p3_[:,0:2] = p3_[:,0:2] / size_scale_
		p3_[:,2] = p3_[:,2] / np.mean(size_scale_)
		p3_[:, 2] *= 500
		p3_[:,2] += resize_factor/2
		
		
		vec_x = np.empty((1,resize_factor))
		vec_y = np.empty((1,resize_factor))
		vec_z = np.empty((1,resize_factor))
		volume_b = np.empty((0,resize_factor,resize_factor,resize_factor))
		#vol_joint_ = np.zeros((64,64,64))
		for jj in xrange(14):
			for kk in xrange(resize_factor):
				vec_x[0,kk] = gaussian(kk, p3_[jj,0],1)
				vec_y[0,kk] = gaussian(kk, p3_[jj, 1], 1)
				vec_z[0,kk] = gaussian(kk, p3_[jj, 2], 1)
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