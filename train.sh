nice -n 10 python train.py \
--structure_string=64 \
--data_split_string_train=S1 \
--data_split_string_test=S1 \
--batch_size=4 \
--joint_prob_max=10 \
--sigma=1 \
--gpu_string=0 \
--load_ckpt_path=./tensor_record/tmp/model1-64.ckpt
