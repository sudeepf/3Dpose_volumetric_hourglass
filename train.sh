nice -n 1 python train.py \
--structure_string=1-2-64 \
--data_split_string_train=S1 \
--data_split_string_test=S9 \
--batch_size=4 \
--joint_prob_max=10 \
--sigma=1 
#--gpu_string=0 \
