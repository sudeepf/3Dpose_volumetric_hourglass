nice -n 10 python train.py \
--structure_string=1-2-4-64 \
--data_split_string_train=S1 \
--data_split_string_test=S1 \
--batch_size=2 \
--joint_prob_max=3 \
--sigma=1 \
--gpu_string=0 \
--learning_rate=1e-4
