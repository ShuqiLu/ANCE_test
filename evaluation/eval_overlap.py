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



checkpoint =  0 
data_type = 0 
test_set = 0 
raw_data_dir = '/home/dihe/Projects/data/raw_data/'
processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-docdev_512'


# sample_list=


# qidmap_path = processed_data_dir+"/qid2offset.pickle"
# pidmap_path = processed_data_dir+"/pid2offset.pickle"

    
# with open(qidmap_path, 'rb') as handle:
#     qidmap = pickle.load(handle)

# with open(pidmap_path, 'rb') as handle:
#     pidmap = pickle.load(handle)

num=900000
# checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_01_05_09/ann_data' +str(num)+'/'
# checkpoint_path2 ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_01_05_09/ann_data'+str(num+10000)+'/'

# checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_12_20_03/ann_data' +'/'
# checkpoint_path_query
# checkpoint_path2 ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_12_20_03/ann_data'+str(110000)+'/'
# checkpoint_path_query2


# checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_01_05_09/ann_data' +'/'
# checkpoint_path_query='/home/dihe/Projects/data/raw_data/eval_exp_12_21_05_90000/ann_data/'
# checkpoint_path2 ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_01_05_09/ann_data'+str(100000)+'/'
# checkpoint_path_query2='/home/dihe/Projects/data/raw_data/eval_exp_01_05_09_100000/ann_data/'

# checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_01_05_09/ann_data'+str(num) +'/'
# checkpoint_path_query='/home/dihe/Projects/data/raw_data/eval_exp_01_05_09_150000/ann_data/'
# checkpoint_path2 ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_01_05_09/ann_data'+str(num+10000)+'/'
# checkpoint_path_query2='/home/dihe/Projects/data/raw_data/eval_exp_01_05_09_160000/ann_data/'


# checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_01_05_09/ann_data'+str(num) +'/'
# checkpoint_path_query='/home/dihe/Projects/data/raw_data/eval_exp_01_05_09_'+str(num)+'/ann_data/'
# checkpoint_path2 ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_01_05_09/ann_data'+str(num+100000)+'/'
# checkpoint_path_query2='/home/dihe/Projects/data/raw_data/eval_exp_01_05_09_'+str(num+100000)+'/ann_data/'



checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_01_05_09/ann_data'+str(num) +'/'
checkpoint_path_query='/home/dihe/Projects/data/raw_data/eval_exp_01_05_09_'+str(num)+'/ann_data/'
checkpoint_path2 ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_21_04_14_01/ann_data'+str(num+10000)+'/'
checkpoint_path_query2='/home/dihe/Projects/data/raw_data/eval_exp_21_04_14_01_'+str(num+10000)+'/ann_data/'



checkpoint = 0

topN=200

dev_query_embedding = []
dev_query_embedding2id = []
passage_embedding = []
passage_embedding2id = []

for i in range(4):
    #try:
    print('???',checkpoint_path_query + "dev_query_"+str(checkpoint)+"__emb_p__data_obj_"+str(i)+".pb")
    with open(checkpoint_path_query + "dev_query_"+str(checkpoint)+"__emb_p__data_obj_"+str(i)+".pb", 'rb') as handle:
        dev_query_embedding.append(pickle.load(handle))
        print('ok1???')
    with open(checkpoint_path_query + "dev_query_"+str(checkpoint)+"__embid_p__data_obj_"+str(i)+".pb", 'rb') as handle:
        dev_query_embedding2id.append(pickle.load(handle))
        print('ok???',2)
    


for i in range(8):
    #try:
    with open(checkpoint_path + "passage_"+str(checkpoint)+"__emb_p__data_obj_"+str(i)+".pb", 'rb') as handle:
        passage_embedding.append(pickle.load(handle))
        print('ok???',3)
    with open(checkpoint_path + "passage_"+str(checkpoint)+"__embid_p__data_obj_"+str(i)+".pb", 'rb') as handle:
        passage_embedding2id.append(pickle.load(handle))
        print('ok???',4)
    # except:
    #     break
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
_, dev_I = cpu_index.search(dev_query_embedding, topN)
# result = EvalDevQuery(dev_query_embedding2id, passage_embedding2id, dev_query_positive_id, dev_I, topN)



topN=200

dev_query_embedding_next = []
dev_query_embedding2id_next = []
passage_embedding_next = []
passage_embedding2id_next = []
for i in range(4):
    #try:
    print('???',checkpoint_path_query2 + "dev_query_"+str(checkpoint)+"__emb_p__data_obj_"+str(i)+".pb")
    with open(checkpoint_path_query2 + "dev_query_"+str(checkpoint)+"__emb_p__data_obj_"+str(i)+".pb", 'rb') as handle:
        dev_query_embedding_next.append(pickle.load(handle))
        print('ok1???')
    with open(checkpoint_path_query2 + "dev_query_"+str(checkpoint)+"__embid_p__data_obj_"+str(i)+".pb", 'rb') as handle:
        dev_query_embedding2id_next.append(pickle.load(handle))
        print('ok???',2)

for i in range(8):
    #try:
    with open(checkpoint_path2 + "passage_"+str(checkpoint)+"__emb_p__data_obj_"+str(i)+".pb", 'rb') as handle:
        passage_embedding_next.append(pickle.load(handle))
        print('ok???',3)
    with open(checkpoint_path2 + "passage_"+str(checkpoint)+"__embid_p__data_obj_"+str(i)+".pb", 'rb') as handle:
        passage_embedding2id_next.append(pickle.load(handle))
        print('ok???',4)
    # except:
    #     break
if (not dev_query_embedding_next) or (not dev_query_embedding2id_next) or (not passage_embedding_next) or not (passage_embedding2id_next):
    print("No data found for checkpoint: ",checkpoint)

dev_query_embedding_next = np.concatenate(dev_query_embedding_next, axis=0)
dev_query_embedding2id_next = np.concatenate(dev_query_embedding2id_next, axis=0)
passage_embedding_next = np.concatenate(passage_embedding_next, axis=0)
passage_embedding2id_next = np.concatenate(passage_embedding2id_next, axis=0)



dim_next = passage_embedding_next.shape[1]
faiss.omp_set_num_threads(16)
cpu_index_next = faiss.IndexFlatIP(dim_next)
cpu_index_next.add(passage_embedding_next)    
_, dev_I_next = cpu_index_next.search(dev_query_embedding_next, topN)
# result = EvalDevQuery(dev_query_embedding2id_next, passage_embedding2id_next, dev_query_positive_id, dev_I, topN)

query_doc={}

for query_idx in range(len(dev_I)): 
    seen_pid = set()
    query_id = dev_query_embedding2id[query_idx]
    

    top_ann_pid = dev_I[query_idx].copy()
    selected_ann_idx = top_ann_pid[:topN]
    
    for idx in selected_ann_idx:
        pred_pid = passage_embedding2id[idx]
        if not pred_pid in seen_pid:
            # this check handles multiple vector per document
            seen_pid.add(pred_pid)
    query_doc[query_id]=seen_pid


query_doc_next={}

for query_idx in range(len(dev_I_next)): 
    seen_pid = set()
    query_id = dev_query_embedding2id_next[query_idx]
    

    top_ann_pid = dev_I_next[query_idx].copy()
    selected_ann_idx = top_ann_pid[:topN]
    
    for idx in selected_ann_idx:
        pred_pid = passage_embedding2id_next[idx]
        if not pred_pid in seen_pid:
            # this check handles multiple vector per document
            seen_pid.add(pred_pid)
    query_doc_next[query_id]=seen_pid


def compute_overlap(a,b):
    assert len(a)==200
    count=0.0
    for  x in a:
        if x in b:
            count+=1
    return count/len(a)

overlap=0.0
for query in query_doc:
    overlap+=compute_overlap(query_doc[query],query_doc_next[query])

print('overlap: ', overlap/len(query_doc))



















