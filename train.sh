nice -n 10 python train.py \
--dataset_dir=/home/capstone/datasets/Human3.6M/Subjects/ \
--structure_string=1-64 \
--data_split_string_train=S1 \
--data_split_string_test=S9 \
--batch_size=1 \
--joint_prob_max=10 \
--sigma=1 \
--load_ckpt_path=/home/capstone/Sudeep/Capstone/3Dpose/tensor_record/tmp/model1-64.ckpt
