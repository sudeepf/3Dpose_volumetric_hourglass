import tensorflow as tf
import numpy as np
import add_summary as summ
from numpy import unravel_index

def compute_precision(prediction, gt, steps, scale, mul_factor, num_joints):
	"""" Given prediction stack, GT coordinates and scale between two """
	pred_cord = get_coordinate(prediction,steps,num_joints)
	pred_cord[:,:,2] /= mul_factor
	pred_cord[:,:,0:2] /= scale[:,:,:]
	err = np.abs(np.linalg.norm(pred_cord-gt))
	print (err)
	return err
	
	
	
def get_coordinate(prediction,steps,num_joints)
	out_shape = np.shape(prediction)
	total_Z = sum(steps)
	pred_ = np.reshape(prediction,(out_shape[0],out_shape[1],out_shape[2],
	                               total_Z, num_joints))
	pred_ = pred_[:,:,:,-1*steps[-1]:,:]
	# Pred_ size is now Batch - X - Y - Z - Joints
	# we need in Batch - Joint - 3(X,Y,Z)
	pred_ = np.rollaxis(pred_,4,1)
	out_shape = np.shape(pred_)
	print (out_shape)
	
	cords = np.zeros(out_shape[0],out_shape[1],3)
	#Would need to itterate over the batch size and joints unfortunately
	for fi, frame in enumerate(pred_[:]):
		for ji, joint in enumerate(frame[:]):
			cords[fi,ji,:] = np.array(unravel_index(joint.argmax(), joint.shape))
	
	return cords