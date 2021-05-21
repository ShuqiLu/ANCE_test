# # Passage ANCE(FirstP) 
# gpu_no=4
# seq_length=512
# model_type=rdot_nll
# tokenizer_type="roberta-base"
# base_data_dir="../data/raw_data/"
# preprocessed_data_dir="${base_data_dir}ann_data_${tokenizer_type}_${seq_length}_dev/"
# job_name="OSPass512"
# pretrained_checkpoint_dir=""


gpu_no=4
seq_length=512
# tokenizer_type="roberta-base-fast"
#tokenizer_type=roberta-base-fast-doceval
# tokenizer_type=roberta-base-fast-doceval_dev
#tokenizer_type=roberta-base-fast-docdev2
model_type=rdot_nll_fairseq_fast
# tokenizer_type=roberta-base-fast-doceval
tokenizer_type=roberta-base-fast-trainqueryeval
base_data_dir="/home/dihe/Projects/data/raw_data/"
preprocessed_data_dir="${base_data_dir}ann_data_${tokenizer_type}_${seq_length}/"
#job_name=eval_exp_12_23_02_400000_dev
# job_name=eval_exp_01_05_09_900000
# pretrained_checkpoint_dir=/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_01_05_09/checkpoint-900000/model.pt
job_name=eval_exp_01_05_09_910000
pretrained_checkpoint_dir=/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_21_04_14_01/check/checkpoint-910000/model.pt

# job_name=eval_exp_01_05_09_100000
# pretrained_checkpoint_dir=/home/dihe/Projects/data/raw_data/exp_01_05_09/checkpoint-100000/model.pt

# job_name=eval_exp_12_21_05_90000
# pretrained_checkpoint_dir=/home/dihe/cudnn_file/recommender_shuqi/exp_12_21_05/save/checkpoint-90000/model.pt


# job_name=eval_exp_01_05_09_150000
# pretrained_checkpoint_dir=/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_01_05_09/checkpoint-150000/model.pt
# job_name=eval_exp_01_05_09_160000
# pretrained_checkpoint_dir=/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_01_05_09/checkpoint-160000/model.pt


# job_name=eval_exp_01_05_09_200000
# pretrained_checkpoint_dir=/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_01_05_09/checkpoint-200000/model.pt

# job_name=eval_exp_01_05_09_210000
# pretrained_checkpoint_dir=/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_01_05_09/checkpoint-210000/model.pt

# job_name=eval_exp_01_05_09_900000
# pretrained_checkpoint_dir=/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_01_05_09/checkpoint-900000/model.pt

# job_name=eval_exp_21_04_14_01_910000
# pretrained_checkpoint_dir=/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_21_04_14_01/checkpoint-910000/model.pt
data_type=0
warmup_steps=3000
per_gpu_train_batch_size=16
gradient_accumulation_steps=1
learning_rate=5e-6

seq_length=128

# # Document ANCE(FirstP) 
# gpu_no=4
# seq_length=512
# model_type=rdot_nll
# tokenizer_type="roberta-base"
# base_data_dir="../data/raw_data/"
# preprocessed_data_dir="${base_data_dir}ann_data_${tokenizer_type}_${seq_length}/"
# job_name="OSDoc512"
# pretrained_checkpoint_dir=""

# # Document ANCE(MaxP)
# gpu_no=4
# seq_length=2048
# model_type=rdot_nll_multi_chunk
# tokenizer_type="roberta-base"
# base_data_dir="../data/raw_data/"
# preprocessed_data_dir="${base_data_dir}ann_data_${tokenizer_type}_${seq_length}/"
# job_name="OSDoc2048"
# pretrained_checkpoint_dir=""

##################################### Inference ################################
model_dir="${base_data_dir}${job_name}/"
model_ann_data_dir="${model_dir}ann_data/"

# initial_data_gen_cmd="\
# python -m torch.distributed.launch --nproc_per_node=$gpu_no ../drivers/run_ann_data_gen_eval.py --training_dir $model_dir \
# --init_model_dir $pretrained_checkpoint_dir --train_model_type $model_type --output_dir $model_ann_data_dir \
# --cache_dir "${model_ann_data_dir}cache/" --data_dir $preprocessed_data_dir --max_seq_length $seq_length \
# --per_gpu_eval_batch_size 64 --topk_training 200 --negative_sample 20 --end_output_num 0 --inference --bpe_vocab_file ../../data/bert-16g-0930/vocab.txt\
# "


initial_data_gen_cmd="\
python -m torch.distributed.launch --nproc_per_node=4 ../drivers/run_ann_data_gen_eval.py --training_dir $model_dir \
--init_model_dir $pretrained_checkpoint_dir --train_model_type $model_type --output_dir $model_ann_data_dir \
--cache_dir "${model_ann_data_dir}cache/" --data_dir $preprocessed_data_dir --max_seq_length $seq_length \
--per_gpu_eval_batch_size 64 --topk_training 200 --negative_sample 20 --end_output_num 0 --inference --bpe_vocab_file ../../data/bert-16g-0930/vocab.txt\
"

echo $initial_data_gen_cmd
eval $initial_data_gen_cmd

# initial_data_gen_cmd="\
# python -m torch.distributed.launch --nproc_per_node=$gpu_no ../drivers/run_ann_data_gen.py --training_dir $model_dir \
# --init_model_dir $pretrained_checkpoint_dir --train_model_type $model_type --output_dir $blob_model_ann_data_dir \
# --cache_dir "${model_ann_data_dir}cache/" --data_dir $preprocessed_data_dir --max_seq_length $seq_length \
# --per_gpu_eval_batch_size 64 --topk_training 200 --negative_sample 20 --end_output_num 0 --inference --bpe_vocab_file /blob/MIND_data/bert-16g-0930/vocab.txt  &"

# echo $initial_data_gen_cmd
# eval $initial_data_gen_cmd



