nice -n 10 python train.py \
--structure_string=1-2-64 \
--data_split_string_train=S1-S0-S5-S6-S7-S8 \
--data_split_string_test=S9 \
--batch_size=2 \
--joint_prob_max=2 \
--sigma=1 \
--gpu_string=0 \
--learning_rate=5e-4
