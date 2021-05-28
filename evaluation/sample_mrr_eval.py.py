import sys
sys.path += ['../utils']
import csv
from tqdm import tqdm 
import collections
import gzip
import pickle
import numpy as np
import faiss
import os
import pytrec_eval
import json
from msmarco_eval import quality_checks_qids, compute_metrics, load_reference


  # location for dumpped query and passage/document embeddings which is output_dir 
#checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_12_02_04/ann_data/' 
# checkpoint =  150000 # embedding from which checkpoint(ie: 200000)
# data_type = 0 # 0 for document, 1 for passage
# test_set = 1 # 0 for dev_set, 1 for eval_set
# raw_data_dir = '/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/'
# processed_data_dir = '/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/ann_data_roberta-base-fast-doc_512'

# checkpoint_path ='/home/dihe/Projects/data/raw_data/exp_12_02_04/ann_data/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-doc_512'

# checkpoint =  0 
# data_type = 0 
# test_set = 1 
# checkpoint_path ='/home/dihe/Projects/data/raw_data/test_roberta_decode_doc/ann_data/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-doc_512'

#--------------------------------------------------------------------------------------
# checkpoint =  0 
# data_type = 0 
# test_set = 0 
# checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_12_19_01/ann_data2/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-docdev_512'

# checkpoint =  0 
# data_type = 0 
# test_set =0
# checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_12_23_02/ann_data400000/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-docdev_512'

# checkpoint =  0 
# data_type = 0 
# test_set = 0 
# checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_12_23_02/ann_data4/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-docdev_512'

# checkpoint =  0 
# data_type = 0 
# test_set = 1 
# checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_01_05_09/ann_data820000/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-docdev2_512'
# # processed_data_dir2 = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-docdev2_512'
# # checkpoint_path2 ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_01_05_09/ann_data820000/'

# processed_data_dir2 = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-doceval_512'
# checkpoint_path2 ='/home/dihe/Projects/data/raw_data/eval_exp_01_05_09_820000/ann_data/'
# query_emb_num=4



checkpoint =  0 
data_type = 0 
test_set = 1 
# checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_01_05_09/ann_data820000/'
# checkpoint_path ='/home/dihe/Projects/data/raw_data/exp_01_05_09/ann_data910000/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-docdev_512'
# processed_data_dir2 = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-docdev2_512'
# checkpoint_path2 ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_01_05_09/ann_data820000/'

#training mrr
# checkpoint_path ='/home/dihe/Projects/data/raw_data/exp_01_05_09/ann_data910000/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-docdev_512'

# processed_data_dir2 = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-trainqueryeval_512'
# checkpoint_path2 ='/home/dihe/Projects/data/raw_data/eval_exp_01_05_09_910000/ann_data/'
# query_emb_num=4




# checkpoint_path ='/home/dihe/Projects/data/raw_data/exp_01_05_09/ann_data910000/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-docdev_512'

# processed_data_dir2 = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-trainqueryeval2_512'
# checkpoint_path2 ='/home/dihe/Projects/data/raw_data/eval_exp_01_05_09_910000/ann_data_sample20q/'
# query_emb_num=4


# checkpoint_path ='/home/dihe/Projects/data/raw_data/exp_21_05_21_01_check/ann_data10000/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-docdev_512'

# processed_data_dir2 = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-trainqueryeval_512'
# checkpoint_path2 ='/home/dihe/Projects/data/raw_data/exp_21_05_21_01_check_10000/ann_data_sampleq/'
# query_emb_num=4

# checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_21_05_21_01/check/ann_data10000/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-docdev_512'

# processed_data_dir2 = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-trainqueryeval2_512'
# checkpoint_path2 ='/home/dihe/Projects/data/raw_data/exp_21_05_21_01_check_10000/ann_data_sample20q/'
# query_emb_num=4

# checkpoint_path ='/home/dihe/Projects/data/raw_data/exp_01_05_09/ann_data900000/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-docdev_512'

# processed_data_dir2 = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-trainqueryeval2_512'
# checkpoint_path2 ='/home/dihe/Projects/data/raw_data/eval_exp_01_05_09_900000/ann_data_sample20q/'
# query_emb_num=4

# checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_21_04_14_01/check/ann_data1000000/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-docdev_512'

# processed_data_dir2 = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-trainqueryeval2_512'
# checkpoint_path2 ='/home/dihe/Projects/data/raw_data/eval_exp_21_04_14_01_1000000/ann_data_sample20q/'
# query_emb_num=4


# checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_21_04_14_01/check/ann_data1000000/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-docdev_512'

# processed_data_dir2 = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-trainqueryeval2_512'
# checkpoint_path2 ='/home/dihe/Projects/data/raw_data/eval_exp_21_04_14_01_1000000/ann_data_sample20q/'
# query_emb_num=4

#-------------------------------
# checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_21_04_14_01/check/ann_data1000000/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-docdev_512'

# processed_data_dir2 = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-trainqueryeval2_512'
# checkpoint_path2 ='/home/dihe/Projects/data/raw_data/eval_exp_21_04_14_01_1000000/ann_data_sample20q/'
# query_emb_num=4


# checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_21_04_14_01/check3/ann_data30000/'
# checkpoint_path2 ='/home/dihe/Projects/data/raw_data/exp_21_04_14_01_check3_30000/ann_data_sample20q/'

# checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_21_04_14_01/check3/ann_data30000/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-docdev_512'

# processed_data_dir2 = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-trainqueryeval2_512'
# checkpoint_path2 ='/home/dihe/Projects/data/raw_data/exp_21_04_14_01_check3_30000/ann_data_sample20q/'
# query_emb_num=4


#dev mrr
# checkpoint_path ='/home/dihe/Projects/data/raw_data/exp_01_05_09/ann_data910000/'
# checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_21_05_21_01/check/ann_data280000/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-docdev_512'

# processed_data_dir2 = processed_data_dir
# checkpoint_path2 =checkpoint_path
# query_emb_num=8

# checkpoint_path ='/home/dihe/Projects/data/raw_data/exp_01_05_09/ann_data1000000/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-docdev_512'

# processed_data_dir2 = processed_data_dir
# checkpoint_path2 =checkpoint_path
# query_emb_num=8

# checkpoint_path ='/home/dihe/Projects/data/raw_data/exp_21_05_21_01_check/ann_data10000/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-docdev_512'

# processed_data_dir2 = processed_data_dir
# checkpoint_path2 =checkpoint_path
# query_emb_num=8

# checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_21_05_21_01/check/ann_data300000/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-docdev_512'

# processed_data_dir2 = processed_data_dir
# checkpoint_path2 =checkpoint_path
# query_emb_num=8

# checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_21_04_14_01/check/ann_data1000000/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-docdev_512'

# processed_data_dir2 = processed_data_dir
# checkpoint_path2 =checkpoint_path
# query_emb_num=8


checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_21_04_14_01/check3/ann_data30000/'
raw_data_dir = '/home/dihe/Projects/data/raw_data/'
processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-docdev_512'

processed_data_dir2 = processed_data_dir
checkpoint_path2 =checkpoint_path
query_emb_num=8


#sample20
# checkpoint_path ='/home/dihe/Projects/data/raw_data/exp_01_05_09/ann_data910000/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-docdev_512'
# processed_data_dir2 = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-trainqueryeval_512'
# checkpoint_path2 ='/home/dihe/Projects/data/raw_data/eval_exp_01_05_09_910000/ann_data/'
# query_emb_num=4
# processed_data_dir_origin='/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-doc_512/'
# processed_data_dir_query_origin='/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-doc_512/'
# ann_path='/home/dihe/Projects/data/raw_data/exp_01_05_09/ann_training_data_0'

# checkpoint_path ='/home/dihe/Projects/data/raw_data/exp_01_05_09/ann_data910000/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-docdev_512'
# processed_data_dir2 = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-trainqueryeval2_512'
# checkpoint_path2 ='/home/dihe/Projects/data/raw_data/eval_exp_01_05_09_910000/ann_data_sample20q/'
# query_emb_num=4
# processed_data_dir_origin='/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-doc_512/'
# processed_data_dir_query_origin='/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-doc_512/'
# ann_path='/home/dihe/Projects/data/raw_data/exp_01_05_09/ann_training_data_0'


# checkpoint_path ='/home/dihe/Projects/data/raw_data/exp_21_05_21_01_check/ann_data100000/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-docdev_512'
# processed_data_dir2 = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-trainqueryeval2_512'
# checkpoint_path2 ='/home/dihe/Projects/data/raw_data/exp_21_05_21_01_check_100000/ann_data_sample20q/'
# query_emb_num=4
# processed_data_dir_origin='/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-doc_512/'
# processed_data_dir_query_origin='/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-doc_512/'
# ann_path='/home/dihe/Projects/data/raw_data/exp_21_05_21_01_check/ann_training_data_0'


# ann_path='/home/dihe/Projects/data/raw_data/exp_21_05_21_01_check/ann_training_data_0'
# checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_21_05_21_01/check/ann_data10000/'
# checkpoint_path2 ='/home/dihe/Projects/data/raw_data/exp_21_05_21_01_check_10000/ann_data_sample20q/'
# # checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_01_05_09/ann_data900000/'
# # checkpoint_path2 ='/home/dihe/Projects/data/raw_data/eval_exp_01_05_09_900000/ann_data_sample20q/'

# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-docdev_512'
# processed_data_dir2 = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-trainqueryeval2_512'
# query_emb_num=4
# processed_data_dir_origin='/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-doc_512/'
# processed_data_dir_query_origin='/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-doc_512/'


# ann_path='/home/dihe/Projects/data/raw_data/exp_01_05_09/ann_training_data_0'
# # checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_21_04_14_01/check/ann_data1000000/'
# # checkpoint_path2 ='/home/dihe/Projects/data/raw_data/eval_exp_21_04_14_01_1000000/ann_data_sample20q/'
# checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_01_05_09/ann_data900000/'
# checkpoint_path2 ='/home/dihe/Projects/data/raw_data/eval_exp_01_05_09_900000/ann_data_sample20q/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-docdev_512'
# processed_data_dir2 = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-trainqueryeval2_512'

# query_emb_num=4
# processed_data_dir_origin='/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-doc_512/'
# processed_data_dir_query_origin='/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-doc_512/'



# ann_path='/home/dihe/Projects/data/raw_data/exp_01_05_09/ann_training_data_0'
# # checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_21_04_14_01/check/ann_data1000000/'
# # checkpoint_path2 ='/home/dihe/Projects/data/raw_data/eval_exp_21_04_14_01_1000000/ann_data_sample20q/'
# checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_21_04_14_01/check3/ann_data30000/'
# checkpoint_path2 ='/home/dihe/Projects/data/raw_data/exp_21_04_14_01_check3_30000/ann_data_sample20q/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-docdev_512'
# processed_data_dir2 = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-trainqueryeval2_512'

# query_emb_num=4
# processed_data_dir_origin='/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-doc_512/'
# processed_data_dir_query_origin='/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-doc_512/'




#sample200
# checkpoint_path ='/home/dihe/Projects/data/raw_data/exp_01_05_09/ann_data910000/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-docdev_512'

# processed_data_dir2 = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-trainqueryeval_512'
# checkpoint_path2 ='/home/dihe/Projects/data/raw_data/eval_exp_01_05_09_910000/ann_data/'
# query_emb_num=4

# processed_data_dir_query_origin=processed_data_dir2
# checkpoint_path_origin='/home/dihe/Projects/data/raw_data/exp_01_05_09/ann_data900000/'
# checkpoint_path_query_origin='/home/dihe/Projects/data/raw_data/eval_exp_01_05_09_900000/ann_data/'



# checkpoint_path ='/home/dihe/Projects/data/raw_data/exp_01_05_09/ann_data910000/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-docdev_512'

# processed_data_dir2 = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-trainqueryeval2_512'
# checkpoint_path2 ='/home/dihe/Projects/data/raw_data/eval_exp_01_05_09_910000/ann_data_sample20q/'
# query_emb_num=4

# processed_data_dir_query_origin=processed_data_dir2
# checkpoint_path_origin='/home/dihe/Projects/data/raw_data/exp_01_05_09/ann_data900000/'
# checkpoint_path_query_origin='/home/dihe/Projects/data/raw_data/eval_exp_01_05_09_900000/ann_data_sample20q/'


# checkpoint_path ='/home/dihe/Projects/data/raw_data/exp_21_05_21_01_check/ann_data100000/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-docdev_512'

# processed_data_dir2 = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-trainqueryeval2_512'
# checkpoint_path2 ='/home/dihe/Projects/data/raw_data/exp_21_05_21_01_check_100000/ann_data_sample20q/'
# query_emb_num=4

# processed_data_dir_query_origin=processed_data_dir2
# checkpoint_path_origin='/home/dihe/Projects/data/raw_data/exp_01_05_09/ann_data900000/'
# checkpoint_path_query_origin='/home/dihe/Projects/data/raw_data/eval_exp_01_05_09_900000/ann_data_sample20q/'



# checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_21_04_14_01/check/ann_data1000000/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-docdev_512'

# processed_data_dir2 = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-trainqueryeval2_512'
# checkpoint_path2 ='/home/dihe/Projects/data/raw_data/eval_exp_21_04_14_01_1000000/ann_data_sample20q/'
# query_emb_num=4

# processed_data_dir_query_origin=processed_data_dir2
# checkpoint_path_origin='/home/dihe/Projects/data/raw_data/exp_01_05_09/ann_data900000/'
# checkpoint_path_query_origin='/home/dihe/Projects/data/raw_data/eval_exp_01_05_09_900000/ann_data_sample20q/'



# checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_21_04_14_01/check3/ann_data30000/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-docdev_512'

# processed_data_dir2 = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-trainqueryeval2_512'
# checkpoint_path2 ='/home/dihe/Projects/data/raw_data/exp_21_04_14_01_check3_30000/ann_data_sample20q/'
# query_emb_num=4

# processed_data_dir_query_origin=processed_data_dir2
# checkpoint_path_origin='/home/dihe/Projects/data/raw_data/exp_01_05_09/ann_data900000/'
# checkpoint_path_query_origin='/home/dihe/Projects/data/raw_data/eval_exp_01_05_09_900000/ann_data_sample20q/'




if data_type == 0:
    topN = 100
else:
    topN = 1000





qidmap_path = processed_data_dir2+"/qid2offset.pickle"
pidmap_path = processed_data_dir+"/pid2offset.pickle"
# if data_type == 0:
#     if test_set == 1:
#         query_path = raw_data_dir+"/docleaderboard-queries.tsv"
#         passage_path = raw_data_dir+"/docleaderboard-top100.tsv"
#     else:
#         query_path = raw_data_dir+"/msmarco-docdev-queries.tsv"
#         passage_path = raw_data_dir+"/msmarco-docdev-top100"
# else:
#     if test_set == 1:
#         query_path = raw_data_dir+"/msmarco-test2019-queries.tsv"
#         passage_path = raw_data_dir+"/msmarco-passagetest2019-top1000.tsv"
#     else:
#         query_path = raw_data_dir+"/queries.dev.small.tsv"
#         passage_path = raw_data_dir+"/top1000.dev.tsv"
    
with open(qidmap_path, 'rb') as handle:
    qidmap = pickle.load(handle)

with open(pidmap_path, 'rb') as handle:
    pidmap = pickle.load(handle)





pidmap_re={}
for item in pidmap:
    assert pidmap[item] not in pidmap_re
    pidmap_re[pidmap[item]]=item #'D'+str(item)

qidmap_re={}
for item in qidmap:
    assert qidmap[item] not in qidmap_re
    qidmap_re[qidmap[item]]=item


def get_reverse_dict(mydict):
    mydict_re={}
    print(mydict)
    for item in mydict:
        assert mydict[item] not in mydict_re
        mydict_re[mydict[item]]=item
    return mydict_re


count_none=0
dev_query_positive_id = {}
#query_positive_id_path = os.path.join(raw_data_dir, "msmarco-doctrain-qrels.tsv")
query_positive_id_path = os.path.join(raw_data_dir, "msmarco-docdev-qrels.tsv")

with open(query_positive_id_path, 'r', encoding='utf8') as f:
    tsvreader = csv.reader(f, delimiter=" ")
    for [topicid,_, docid, rel] in tsvreader:
        topicid = int(topicid)
        docid = int(docid[1:])
        if topicid not in dev_query_positive_id:
            dev_query_positive_id[topicid] = {}
        dev_query_positive_id[topicid][docid] = int(rel)
        assert len(dev_query_positive_id[topicid])==1
        if docid not in pidmap:
            count_none+=1
print('count_none: ',count_none)



# qset = set()
# with gzip.open(query_path, 'rt', encoding='utf-8') if query_path[-2:] == "gz" else open(query_path, 'rt', encoding='utf-8') as f:
#     tsvreader = csv.reader(f, delimiter="\t")
#     for [qid, query] in tsvreader:
#         qset.add(qid)

# bm25 = collections.defaultdict(set)
# with gzip.open(passage_path, 'rt', encoding='utf-8') if passage_path[-2:] == "gz" else open(passage_path, 'rt', encoding='utf-8') as f:
#     for line in tqdm(f):
#         if data_type == 0:
#             [qid, Q0, pid, rank, score, runstring] = line.split(' ')
#             pid = pid[1:]
#         else:
#             [qid, pid, query, passage] = line.split("\t")
#             #print('???',qid)
#         if qid in qset and int(qid) in qidmap:
#             bm25[qidmap[int(qid)]].add(pidmap[int(pid)])
#         # else:
#         #     print('???',qid,qid in qset)

# #assert 1==0
# print("number of queries with " +str(topN) + " BM25 passages:", len(bm25))

def get_sample20():

    
    #train_queries={}
    with open(processed_data_dir_query_origin+"/qid2offset_train.pickle", 'rb') as handle:
        qidmap_origin = pickle.load(handle)

    with open(processed_data_dir_query_origin+"/pid2offset.pickle", 'rb') as handle:
        pidmap_origin = pickle.load(handle)
    qidmap_origin_re={}
    for item in qidmap_origin:
        assert qidmap_origin[item] not in qidmap_origin_re
        qidmap_origin_re[qidmap_origin[item]]=item

    pidmap_origin_re={}
    for item in pidmap_origin:
        assert pidmap_origin[item] not in pidmap_origin_re
        pidmap_origin_re[pidmap_origin[item]]=item

    

    train_q_sample20={}
    with open(ann_path, 'r') as f:
        ann_training_data = f.readlines()

    # aligned_size = (len(ann_training_data) // 8) * 8
    # ann_training_data = ann_training_data[:aligned_size]


    # passage_embedding2id=[]
    # for i in range(8):
    #     # with open(checkpoint_path + "passage_"+str(checkpoint)+"__emb_p__data_obj_"+str(i)+".pb", 'rb') as handle:
    #     #     passage_embedding.append(pickle.load(handle))
    #     #     print('ok???',3,i)
    #     with open(checkpoint_path_origin + "passage_"+str(checkpoint)+"__embid_p__data_obj_"+str(i)+".pb", 'rb') as handle:
    #         passage_embedding2id.append(pickle.load(handle))
    #         print('ok???',4,i)

    # #passage_embedding2id = np.concatenate(passage_embedding2id, axis=0)

    # dev_query_embedding2id=[]
    # for i in range(4):
    #     print('???',checkpoint_path_query_origin + "dev_query_"+str(checkpoint)+"__emb_p__data_obj_"+str(i)+".pb")
    #     #with open(checkpoint_path2 + "dev_query_"+str(checkpoint)+"__emb_p__data_obj_"+str(i)+".pb", 'rb') as handle:
    #     with open(checkpoint_path_query_origin + "dev_query_"+str(checkpoint)+"__embid_p__data_obj_"+str(i)+".pb", 'rb') as handle:
    #         dev_query_embedding2id.append(pickle.load(handle))
    #         print('ok???',2)

    # if (not dev_query_embedding2id) or not (passage_embedding2id):
    #     print("No data found for checkpoint: ",checkpoint)

    # passage_embedding2id = np.concatenate(passage_embedding2id, axis=0)
    # dev_query_embedding2id=np.concatenate(dev_query_embedding2id, axis=0)

    # passage_embedding2id2_r=reverse_dict(passage_embedding2id2)
    # dev_query_embedding2id_r=reverse_dict(dev_query_embedding2id)

    for line in ann_training_data:
        line_arr=line.strip().split('\t')
        # qid = qidmap_origin_re[dev_query_embedding2id[int(line_arr[0])]]
        # if qid in dev_query_positive_id:
        #     pos_pid = pidmap_re[passage_embedding2id[int(line_arr[1])]]
        #     neg_pids = line_arr[2].split(',')
        #     neg_pids = [pidmap_re[passage_embedding2id[int(neg_pid)]] for neg_pid in neg_pids]
        #     train_q_sample20[qid]=neg_pids+[pos_pid]
        qid = qidmap_origin_re[int(line_arr[0])]
        if qid in dev_query_positive_id:
            pos_pid = pidmap_origin_re[int(line_arr[1])]
            neg_pids = line_arr[2].split(',')
            neg_pids = [pidmap_origin_re[int(neg_pid)] for neg_pid in neg_pids]
            train_q_sample20[qid]=neg_pids[:20]+[pos_pid]
        else:
            assert 1==0

    return train_q_sample20





 
def get_sample200():


    #dev_query_positive_id[topicid]

    with open(processed_data_dir_query_origin+"/qid2offset.pickle", 'rb') as handle:
        qidmap_origin = pickle.load(handle)
    qidmap_origin_re={}
    for item in qidmap_origin:
        assert qidmap_origin[item] not in qidmap_origin_re
        qidmap_origin_re[qidmap_origin[item]]=item

    dev_query_embedding=[]
    dev_query_embedding2id=[]
    train_q_sample200={}
    for i in range(4):
        #try:
        print('???',checkpoint_path_query_origin + "dev_query_"+str(checkpoint)+"__emb_p__data_obj_"+str(i)+".pb")
        with open(checkpoint_path_query_origin + "dev_query_"+str(checkpoint)+"__emb_p__data_obj_"+str(i)+".pb", 'rb') as handle:
            dev_query_embedding.append(pickle.load(handle))
            print('ok1???')
        with open(checkpoint_path_query_origin + "dev_query_"+str(checkpoint)+"__embid_p__data_obj_"+str(i)+".pb", 'rb') as handle:
            dev_query_embedding2id.append(pickle.load(handle))
            print('ok???',2)

    
    passage_embedding=[]
    passage_embedding2id=[]
    for i in range(8):
        with open(checkpoint_path_origin + "passage_"+str(checkpoint)+"__emb_p__data_obj_"+str(i)+".pb", 'rb') as handle:
            passage_embedding.append(pickle.load(handle))
            print('ok???',3,i)
        with open(checkpoint_path_origin + "passage_"+str(checkpoint)+"__embid_p__data_obj_"+str(i)+".pb", 'rb') as handle:
            passage_embedding2id.append(pickle.load(handle))
            print('ok???',4,i)

    if (not dev_query_embedding) or (not dev_query_embedding2id) or (not passage_embedding) or not (passage_embedding2id):
        print("No data found for checkpoint: ",checkpoint)


    dev_query_embedding = np.concatenate(dev_query_embedding, axis=0)
    dev_query_embedding2id = np.concatenate(dev_query_embedding2id, axis=0)
    passage_embedding = np.concatenate(passage_embedding, axis=0)
    passage_embedding2id = np.concatenate(passage_embedding2id, axis=0)


    dim = passage_embedding.shape[1]
    faiss.omp_set_num_threads(16)
    cpu_index = faiss.IndexFlatIP(dim)
    cpu_index.add(passage_embedding)    
    _, dev_I = cpu_index.search(dev_query_embedding, 200)

    #dev_query_embedding2id_r=get_reverse_dict(dev_query_embedding2id)

    #for item in train_queries:
    query_list=[]
    # for i,query_idx in enumerate(dev_query_embedding2id):
    for i,query_idx in enumerate(range(len(dev_I))): 
        query_id = qidmap_origin_re[dev_query_embedding2id[query_idx]]
        selected_ann_idx=dev_I[query_idx]
        if query_id in dev_query_positive_id:
            train_q_sample200[query_id]=[]
            pos_id=list(dev_query_positive_id[query_id].keys())[0]
            for idx in selected_ann_idx:
                pred_pid = pidmap_re[passage_embedding2id[idx]]
                train_q_sample200[query_id].append(pred_pid)

            train_q_sample200[query_id]=train_q_sample200[query_id][:20]

            if pos_id not in  train_q_sample200[query_id]:
                train_q_sample200[query_id]+=[pos_id] 
        # if i<5:
        #     print(query_id,dev_query_embedding2id[query_idx],query_idx)
        #     query_list.append(query_id)

    #print([train_q_sample200[x] for x in query_list])

        
    return train_q_sample200






    

def get_all(passage_embedding,passage_embedding2id):

    for i in range(8):
        #try:
        # print('???',checkpoint_path2 + "dev_query_"+str(checkpoint)+"__emb_p__data_obj_"+str(i)+".pb")
        # with open(checkpoint_path2 + "dev_query_"+str(checkpoint)+"__emb_p__data_obj_"+str(i)+".pb", 'rb') as handle:
        #     dev_query_embedding.append(pickle.load(handle))
        #     print('ok1???')
        # with open(checkpoint_path2 + "dev_query_"+str(checkpoint)+"__embid_p__data_obj_"+str(i)+".pb", 'rb') as handle:
        #     dev_query_embedding2id.append(pickle.load(handle))
        #     print('ok???',2)
        with open(checkpoint_path + "passage_"+str(checkpoint)+"__emb_p__data_obj_"+str(i)+".pb", 'rb') as handle:
            passage_embedding.append(pickle.load(handle))
            print('ok???',3,i)
        with open(checkpoint_path + "passage_"+str(checkpoint)+"__embid_p__data_obj_"+str(i)+".pb", 'rb') as handle:
            passage_embedding2id.append(pickle.load(handle))
            print('ok???',4,i)
        # except:
        #     break
    if (not passage_embedding) or not (passage_embedding2id):
        print("No data found for checkpoint: ",checkpoint)
    passage_embedding = np.concatenate(passage_embedding, axis=0)
    passage_embedding2id = np.concatenate(passage_embedding2id, axis=0)
    return passage_embedding,passage_embedding2id



def convert_to_string_id(result_dict):
    string_id_dict = {}

    # format [string, dict[string, val]]
    for k, v in result_dict.items():
        _temp_v = {}
        for inner_k, inner_v in v.items():
            _temp_v[str(inner_k)] = inner_v

        string_id_dict[str(k)] = _temp_v

    return string_id_dict

def EvalDevQuery(query_embedding2id, passage_embedding2id, qidmap_re,pidmap_re, dev_query_positive_id,I_nearest_neighbor,topN,bm25=None):
    prediction = {} #[qid][docid] = docscore, here we use -rank as score, so the higher the rank (1 > 2), the higher the score (-1 > -2)

    #w=open('result_eval.txt','w')
    total = 0
    labeled = 0
    Atotal = 0
    Alabeled = 0
    qids_to_ranked_candidate_passages = {} 
    mrr=0.0
    mycount=0
    for query_idx in range(len(I_nearest_neighbor)): 
        seen_pid = set()
        query_id = qidmap_re[query_embedding2id[query_idx]]

        if bm25 and query_id not in bm25:
            #assert 1==0
            continue

        prediction[query_id] = {}

        top_ann_pid = I_nearest_neighbor[query_idx].copy()
        selected_ann_idx = top_ann_pid[:topN]
        #print('???',topN)

        #if train_q_sample20 !=None:

        rank = 0
        flag=0

        if query_id in qids_to_ranked_candidate_passages:
            assert 1==0,"query not in"
            pass    
        else:
            # By default, all PIDs in the list of 1000 are 0. Only override those that are given
            tmp = [0] * 1000
            qids_to_ranked_candidate_passages[query_id] = tmp
                
        mycount+=1
        for idx in selected_ann_idx:
            pred_pid = pidmap_re[passage_embedding2id[idx]]
            
            if not pred_pid in seen_pid:
                # this check handles multiple vector per document
                qids_to_ranked_candidate_passages[query_id][rank]=pred_pid
                #w.write(str(query_id)+'\t'+str(pred_pid)+'\t'+str(rank+1)+'\n')
                # assert len(dev_query_positive_id[query_id]) ==1
                # for item in dev_query_positive_id[query_id]:
                #     assert item in pidmap

                assert pred_pid in pidmap
                if pred_pid in dev_query_positive_id[query_id]:
                    mrr += 1/(rank + 1)
                    flag=1
                    #print('rank: ',rank)
                    


                Atotal += 1
                if pred_pid not in dev_query_positive_id[query_id]:
                    Alabeled += 1
                if rank < 10:
                    total += 1
                    if pred_pid not in dev_query_positive_id[query_id]:
                        labeled += 1
                rank += 1
                prediction[query_id][pred_pid] = -rank
                seen_pid.add(pred_pid)

        #assert rank!=0, "pos not in"
    # w.close()
    evaluator = pytrec_eval.RelevanceEvaluator(
        convert_to_string_id(dev_query_positive_id), {'map_cut', 'ndcg_cut', 'recip_rank','recall'})
    eval_query_cnt = 0
    result = evaluator.evaluate(convert_to_string_id(prediction))


    print('???',mrr/mycount,mycount,mrr,len(I_nearest_neighbor))
    
    qids_to_relevant_passageids = {}
    for qid in dev_query_positive_id:
        qid = int(qid)
        if qid in qids_to_relevant_passageids:
            pass
        else:
            qids_to_relevant_passageids[qid] = []
            for pid in dev_query_positive_id[qid]:
                #assert pid>0
                if pid>0:
                    qids_to_relevant_passageids[qid].append(pid)
            
    if data_type == 0:
        MaxMRRRank=100
    else:
        MaxMRRRank=10


    ms_mrr = compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages,MaxMRRRank=MaxMRRRank)
    # ms_mrr = compute_metrics(dev_query_positive_id, qids_to_ranked_candidate_passages,MaxMRRRank=MaxMRRRank)
    # print('???',ms_mrr)

    ndcg = 0
    Map = 0
    mrr = 0
    recall = 0
    recall_1000 = 0

    for k in result.keys():
        eval_query_cnt += 1
        ndcg += result[k]["ndcg_cut_10"]
        Map += result[k]["map_cut_10"]
        mrr += result[k]["recip_rank"]
        recall += result[k]["recall_"+str(topN)]

    final_ndcg = ndcg / eval_query_cnt
    final_Map = Map / eval_query_cnt
    final_mrr = mrr / eval_query_cnt
    final_recall = recall / eval_query_cnt
    hole_rate = labeled/total
    Ahole_rate = Alabeled/Atotal

    return final_ndcg, eval_query_cnt, final_Map, final_mrr, final_recall, hole_rate, ms_mrr, Ahole_rate, result, prediction


dev_query_embedding = []
dev_query_embedding2id = []
passage_embedding = []
passage_embedding2id = []


for i in range(query_emb_num):
    #try:
    print('???',checkpoint_path2 + "dev_query_"+str(checkpoint)+"__emb_p__data_obj_"+str(i)+".pb")
    with open(checkpoint_path2 + "dev_query_"+str(checkpoint)+"__emb_p__data_obj_"+str(i)+".pb", 'rb') as handle:
        dev_query_embedding.append(pickle.load(handle))
        print('ok1???')
    with open(checkpoint_path2 + "dev_query_"+str(checkpoint)+"__embid_p__data_obj_"+str(i)+".pb", 'rb') as handle:
        dev_query_embedding2id.append(pickle.load(handle))
        print('ok???',2)



passage_embedding,passage_embedding2id=get_all(passage_embedding,passage_embedding2id)

# sample20 =get_sample200(passage_embedding)

# passage_embedding,passage_embedding2id =get_sample200(passage_embedding)

if (not dev_query_embedding) or (not dev_query_embedding2id):
    print("No data found for checkpoint: ",checkpoint)

dev_query_embedding = np.concatenate(dev_query_embedding, axis=0)
dev_query_embedding2id = np.concatenate(dev_query_embedding2id, axis=0)





##reranking
# sample20 =get_sample20()

# sample20 =get_sample200()
# pidmap_t = collections.defaultdict(list)
# for i in range(len(passage_embedding2id)):
#     pidmap_t[pidmap_re[passage_embedding2id[i]]].append(i)  # abs pos(key) to rele pos(val)
# all_dev_I = []
# for i,qid in enumerate(range(len(dev_query_embedding2id))):
#     qid_r=qidmap_re[dev_query_embedding2id[qid]]
#     p_set = []
#     p_set_map = {}
#     if qid_r not in sample20:
#         print('no')
        
#     else:
#         #print('yes')
#         count = 0
#         for k,pid in enumerate(sample20[qid_r]):
#             if pid in pidmap_t:
#                 for val in pidmap_t[pid]:
#                     p_set.append(passage_embedding[val])
#                     p_set_map[count] = val # new rele pos(key) to old rele pos(val)
#                     count += 1
#             else:
#                 print(pid,"not in passages")
#     #print('???len(p_set)',len(p_set))
#     if len(p_set)==0:
#         all_dev_I.append([-1]*10)
#     else:
        
#         dim = passage_embedding.shape[1]
#         faiss.omp_set_num_threads(16)
#         cpu_index = faiss.IndexFlatIP(dim)
#         p_set =  np.asarray(p_set)
#         cpu_index.add(p_set)    
#         _, dev_I = cpu_index.search(dev_query_embedding[i:i+1], len(p_set))
#         # if i<5:
#         #     print(sample20[qid_r],qid_r,dev_query_embedding2id[qid],qid)
#         # if i<5:
#         #     print(dev_I,dev_query_positive_id[qid_r])
#         for j in range(len(dev_I[0])):
#             dev_I[0][j] = p_set_map[dev_I[0][j]]
#         # if i<5:
#         #     print(dev_I,dev_query_positive_id[qid_r])
#         #     print([pidmap_re[passage_embedding2id[x]] for x in dev_I[0]])
#         #     print('-----------------------')
#         all_dev_I.append(dev_I[0])
# print(len(sample20),len(all_dev_I))
# result = EvalDevQuery(dev_query_embedding2id, passage_embedding2id, qidmap_re,pidmap_re, dev_query_positive_id, all_dev_I, 10,bm25=sample20)
# final_ndcg, eval_query_cnt, final_Map, final_mrr, final_recall, hole_rate, ms_mrr, Ahole_rate, metrics, prediction = result
# print("Reranking Results for checkpoint "+str(checkpoint))
# print("Reranking NDCG@10:" + str(final_ndcg))
# print("Reranking map@10:" + str(final_Map))
# print("Reranking pytrec_mrr:" + str(final_mrr))
# print("Reranking recall@"+str(topN)+":" + str(final_recall))
# print("Reranking hole rate@10:" + str(hole_rate))
# print("Reranking hole rate:" + str(Ahole_rate))
# print("Reranking ms_mrr:" + str(ms_mrr))












#full ranking
dim = passage_embedding.shape[1]
faiss.omp_set_num_threads(16)
cpu_index = faiss.IndexFlatIP(dim)
cpu_index.add(passage_embedding)    
_, dev_I = cpu_index.search(dev_query_embedding, topN)
#print('???',dev_I[:10])
result = EvalDevQuery(dev_query_embedding2id, passage_embedding2id, qidmap_re,pidmap_re , dev_query_positive_id,dev_I, 100)

final_ndcg, eval_query_cnt, final_Map, final_mrr, final_recall, hole_rate, ms_mrr, Ahole_rate, metrics, prediction = result
print("Results for checkpoint "+str(checkpoint))
print("NDCG@10:" + str(final_ndcg))
print("map@10:" + str(final_Map))
print("pytrec_mrr:" + str(final_mrr))
print("recall@"+str(topN)+":" + str(final_recall))
print("hole rate@10:" + str(hole_rate))
print("hole rate:" + str(Ahole_rate))
print("ms_mrr:" + str(ms_mrr))








