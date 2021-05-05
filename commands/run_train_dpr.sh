gpu_no=4

# model type
model_type="dpr_fast"
seq_length=256
triplet="--triplet --optimizer lamb" # set this to empty for non triplet model

# hyper parameters
train_batch_size=8
gradient_accumulation_steps=2
learning_rate=1e-5
warmup_steps=1000

# input/output directories
preprocessed_data_dir="../../data/raw_data/QA_NQ_data/" 
job_name="ann_NQ_test"
model_dir="./"
model_ann_data_dir="${model_dir}ann_data/"
pretrained_checkpoint_dir="../../data/model_temp/dpr_biencoder.37"

blob_output_dir='../../data/raw_data/exp_21_04_20_01/'
blob_ann_dir="${blob_output_dir}ann_data/"

train_cmd="\
python -m torch.distributed.launch --nproc_per_node=$gpu_no ../drivers/run_ann_dpr.py --model_type $model_type \
--model_name_or_path $pretrained_checkpoint_dir --task_name MSMarco $triplet --data_dir $preprocessed_data_dir \
--ann_dir $model_ann_data_dir --max_seq_length $seq_length --per_gpu_train_batch_size=$train_batch_size \
--gradient_accumulation_steps $gradient_accumulation_steps --learning_rate $learning_rate --output_dir $model_dir \
--warmup_steps $warmup_steps --logging_steps 100 --save_steps 10000 --log_dir "${model_dir}/log/" --blob_ann_dir ${blob_ann_dir} --blob_output_dir ${blob_output_dir} \
"

echo $train_cmd
eval $train_cmd

# echo "copy current script to model directory"
# sudo cp $0 $model_dir