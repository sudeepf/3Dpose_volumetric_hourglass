nice -n 10 python train.py \
--structure_string=2-4-64 \
--data_split_string_train=S1-S0-S5-S6-S7-S8 \
--data_split_string_test=S1 \
<<<<<<< HEAD
--batch_size=4 \
--joint_prob_max=3 \
--sigma=1 \
--gpu_string=0-1 \
--learning_rate=1e-4 
=======
--batch_size=8 \
--joint_prob_max=1 \
--sigma=1 \
--gpu_string=0-1 \
--learning_rate=1e-4
>>>>>>> e22accb20ef4df1ffd3d7cf6eaa35b4c3ecf7782

