nice -n 1 python train.py \
--structure_string=1-64 \
--data_split_string_train=S1 \
--data_split_string_test=S1 \
--batch_size=1 \
--joint_prob_max=10 \
--sigma=1 
#--gpu_string=0 \
