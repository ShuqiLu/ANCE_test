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

checkpoint =  0 
data_type = 0 
test_set = 1 
checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_01_05_09/ann_data820000/'
raw_data_dir = '/home/dihe/Projects/data/raw_data/'
processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-docdev2_512'
# processed_data_dir2 = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-docdev2_512'
# checkpoint_path2 ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_01_05_09/ann_data820000/'

processed_data_dir2 = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-doceval_512'
checkpoint_path2 ='/home/dihe/Projects/data/raw_data/eval_exp_01_05_09_820000/ann_data/'
query_emb_num=4

# processed_data_dir2 = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-doceval_dev_512'
# checkpoint_path2 ='/home/dihe/Projects/data/raw_data/eval_exp_01_05_09_820000_dev/ann_data/'
# query_emb_num=4


# checkpoint =  0 
# data_type = 1 
# test_set = 0 
# checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_12_21_05/ann_data2/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast_512'


# checkpoint =  0 
# data_type = 1 
# test_set = 0 
# checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_02_03_02/ann_data/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-passsmall5_512'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir2 = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-passsmall5_512'
# checkpoint_path2 ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_02_03_02/ann_data/'

# checkpoint =  0 
# data_type = 0
# test_set = 0 
# checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_12_23_02/ann_data400000/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-docdev_512'
# processed_data_dir2 = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-doceval_dev_512'
# checkpoint_path2 ='/home/dihe/Projects/data/raw_data/eval_exp_12_23_02_400000_dev/ann_data/'

#820

# checkpoint =  0 
# data_type = 1 
# test_set = 0 
# checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_12_23_08/ann_data2/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast_512'

# checkpoint =  0 
# data_type = 1 
# test_set = 0 
# checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_12_20_03/ann_data2/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast_512'



# checkpoint =  0 
# data_type = 1 
# test_set = 0 
# checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/exp_12_02_14_01/save/ann_data/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast_512'


# checkpoint =  0 
# data_type = 1 
# test_set = 0 
# checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/exp_01_07_09/save/ann_data/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast_512'

# checkpoint =  0 
# data_type = 1 
# test_set = 0 
# checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_12_23_08/ann_data/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast_512'

# checkpoint =  0 
# data_type = 1 
# test_set = 1 
# checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_11_11_01/ann_data3/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-passtest_512'
#--------------------------------------------------------------------------------------------

# checkpoint =  180000 
# data_type = 1 
# test_set = 1 
# checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_12_02_02/ann_data/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast_512'


# checkpoint =  210000 
# data_type = 1 
# test_set = 1 
# checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_12_02_01/ann_data/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast_512'

# checkpoint =  0 
# data_type = 1 
# test_set = 0
# checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_12_11_03/ann_data2/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast_512'



if data_type == 0:
    topN = 100
else:
    topN = 1000
# dev_query_positive_id = {}
# query_positive_id_path = os.path.join(processed_data_dir, "dev-qrel.tsv")

# with open(query_positive_id_path, 'r', encoding='utf8') as f:
#     tsvreader = csv.reader(f, delimiter="\t")
#     for [topicid, docid, rel] in tsvreader:
#         topicid = int(topicid)
#         docid = int(docid)
#         if topicid not in dev_query_positive_id:
#             dev_query_positive_id[topicid] = {}
#         dev_query_positive_id[topicid][docid] = int(rel)




qidmap_path = processed_data_dir2+"/qid2offset.pickle"
pidmap_path = processed_data_dir+"/pid2offset.pickle"
if data_type == 0:
    if test_set == 1:
        query_path = raw_data_dir+"/docleaderboard-queries.tsv"
        passage_path = raw_data_dir+"/docleaderboard-top100.tsv"
    else:
        query_path = raw_data_dir+"/msmarco-docdev-queries.tsv"
        passage_path = raw_data_dir+"/msmarco-docdev-top100"
else:
    if test_set == 1:
        query_path = raw_data_dir+"/msmarco-test2019-queries.tsv"
        passage_path = raw_data_dir+"/msmarco-passagetest2019-top1000.tsv"
    else:
        query_path = raw_data_dir+"/queries.dev.small.tsv"
        passage_path = raw_data_dir+"/top1000.dev.tsv"
    
with open(qidmap_path, 'rb') as handle:
    qidmap = pickle.load(handle)

with open(pidmap_path, 'rb') as handle:
    pidmap = pickle.load(handle)

qidmap_re={}
for item in qidmap:
    assert qidmap[item] not in qidmap_re
    qidmap_re[qidmap[item]]=item

pidmap_re={}
for item in pidmap:
    assert pidmap[item] not in pidmap_re
    pidmap_re[pidmap[item]]='D'+str(item)



qset = set()
with gzip.open(query_path, 'rt', encoding='utf-8') if query_path[-2:] == "gz" else open(query_path, 'rt', encoding='utf-8') as f:
    tsvreader = csv.reader(f, delimiter="\t")
    for [qid, query] in tsvreader:
        qset.add(qid)

bm25 = collections.defaultdict(set)
with gzip.open(passage_path, 'rt', encoding='utf-8') if passage_path[-2:] == "gz" else open(passage_path, 'rt', encoding='utf-8') as f:
    for line in tqdm(f):
        if data_type == 0:
            [qid, Q0, pid, rank, score, runstring] = line.split(' ')
            pid = pid[1:]
        else:
            [qid, pid, query, passage] = line.split("\t")
            #print('???',qid)
        if qid in qset and int(qid) in qidmap:
            bm25[qidmap[int(qid)]].add(pidmap[int(pid)])
        # else:
        #     print('???',qid,qid in qset)

#assert 1==0
print("number of queries with " +str(topN) + " BM25 passages:", len(bm25))




def convert_to_string_id(result_dict):
    string_id_dict = {}

    # format [string, dict[string, val]]
    for k, v in result_dict.items():
        _temp_v = {}
        for inner_k, inner_v in v.items():
            _temp_v[str(inner_k)] = inner_v

        string_id_dict[str(k)] = _temp_v

    return string_id_dict

def EvalDevQuery(query_embedding2id, passage_embedding2id, qidmap_re,pidmap_re, I_nearest_neighbor,topN):
    prediction = {} #[qid][docid] = docscore, here we use -rank as score, so the higher the rank (1 > 2), the higher the score (-1 > -2)

    w=open('result_eval.txt','w')
    total = 0
    labeled = 0
    Atotal = 0
    Alabeled = 0
    qids_to_ranked_candidate_passages = {} 
    for query_idx in range(len(I_nearest_neighbor)): 
        seen_pid = set()
        query_id = qidmap_re[query_embedding2id[query_idx]]
        prediction[query_id] = {}

        top_ann_pid = I_nearest_neighbor[query_idx].copy()
        selected_ann_idx = top_ann_pid[:topN]
        rank = 0
        
        if query_id in qids_to_ranked_candidate_passages:
            pass    
        else:
            # By default, all PIDs in the list of 1000 are 0. Only override those that are given
            tmp = [0] * 1000
            qids_to_ranked_candidate_passages[query_id] = tmp
                
        for idx in selected_ann_idx:
            pred_pid = pidmap_re[passage_embedding2id[idx]]
            
            if not pred_pid in seen_pid:
                # this check handles multiple vector per document
                qids_to_ranked_candidate_passages[query_id][rank]=pred_pid
                w.write(str(query_id)+'\t'+str(pred_pid)+'\t'+str(rank+1)+'\n')
                # Atotal += 1
                # if pred_pid not in dev_query_positive_id[query_id]:
                #     Alabeled += 1
                # if rank < 10:
                #     total += 1
                #     if pred_pid not in dev_query_positive_id[query_id]:
                #         labeled += 1
                rank += 1
                #prediction[query_id][pred_pid] = -rank
                seen_pid.add(pred_pid) 
    w.close()



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



#full ranking
dim = passage_embedding.shape[1]
faiss.omp_set_num_threads(16)
cpu_index = faiss.IndexFlatIP(dim)
cpu_index.add(passage_embedding)    
_, dev_I = cpu_index.search(dev_query_embedding, topN)
EvalDevQuery(dev_query_embedding2id, passage_embedding2id, qidmap_re,pidmap_re , dev_I, topN)









