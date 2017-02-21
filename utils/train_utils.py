import tensorflow as tf
import numpy as np
import add_summary as summ
from numpy import unravel_index
import matplotlib.pyplot as plt

def compute_precision(prediction, gt, steps, mul_factor, num_joints):
	"""" Given prediction stack, GT coordinates and scale between two """
	predictions_cord  = get_coordinate(prediction,steps,num_joints)
	joint_wise_error = np.zeros((num_joints))
	for ii, pred_cord in enumerate(predictions_cord):
		gt_ = gt[ii]
		
		
		pred_cord = pred_cord.astype(float)
		# Get Root Joints ie Hips
		RJ1_pred = pred_cord[8,:]
		RJ2_pred = pred_cord[11,:]
		RJ1_gt = gt_[8,:]
		RJ2_gt = gt_[11,:]

		
		
		# Get mean of pose
		MR_gt = (RJ1_gt + RJ2_gt)/2
		MR_pred = (RJ1_pred + RJ2_pred)/2
		
		# Allign mean of pred to mean of GT
		
		pred_cord -=  MR_pred
		gt_ -= MR_gt
		
		#Get Root limb length
		RL_pred = np.linalg.norm(RJ1_pred[0:2]-RJ2_pred[0:2])
		RL_gt = np.linalg.norm(RJ1_gt[0:2] - RJ2_gt[0:2])
		
		
		#Get scale from limb length
		scale_ = RL_gt/RL_pred
		
		pred_cord[:,2] /= mul_factor
		pred_cord[:,:] *= scale_
		
		#plt.scatter(x=pred_cord[:, 0], y=pred_cord[:, 1], c='r')
		#plt.scatter(x=gt_[:, 0], y=gt_[:, 1], c='b')
		#plt.show()
		
		for jj in xrange(num_joints):
			joint_wise_error[jj] += (np.linalg.norm(pred_cord[jj]-gt_[jj]))
	
	
	return joint_wise_error/np.shape(prediction)[0]
	
	
	
def get_coordinate(prediction,steps,num_joints):
	out_shape = np.shape(prediction)
	total_Z = sum(steps)
	pred_ = np.reshape(prediction,(out_shape[0],out_shape[1],out_shape[2],
	                               total_Z, num_joints))
	#print(np.shape(pred_))
	#plt.imshow(np.sum(pred_[0, :, :, 0, :], axis=2))
	#plt.show()
	pred_ = pred_[:,:,:,-1*steps[-1]:,:]
	# Pred_ size is now Batch - X - Y - Z - Joints
	# we need in Batch - Joint - 3(X,Y,Z)
	pred_ = np.rollaxis(pred_,4,1)
	pred_ = np.swapaxes(pred_,2,3)
	out_shape = np.shape(pred_)
	
	cords = np.zeros((out_shape[0],out_shape[1],3))
	#Would need to itterate over the batch size and joints unfortunately
	for fi, frame in enumerate(pred_[:]):
		for ji, joint in enumerate(frame[:]):
			cords[fi,ji,:] = np.array(unravel_index(joint.argmax(), joint.shape))
	
	#plt.scatter(x=cords[0,:, 0], y=cords[0,:, 1], c='b')
	#plt.show()
	return cords