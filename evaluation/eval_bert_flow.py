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
# test_set = 0 
# checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/exp_01_05_09/ann_data910000/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-docdev_512'

#-----------------------------------------------------------------------------

# exp=sys.argv[1] 
# model_num=sys.argv[2] 

# checkpoint =  0 
# data_type = 0 
# test_set = 0 
# checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/'+str(exp)+'/ann_data'+str(model_num)+'/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-docdev_512'



# checkpoint =  0 
# data_type = 0 
# test_set = 0 
# checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/'+str()+'/ann_data910000/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-docdev2_512'

#-----------------------------------------------------------------------------

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



# exp=sys.argv[1] 
# model_num=sys.argv[2] 

# checkpoint =  0 
# data_type = 1 
# test_set = 0 
# checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/'+str(exp)+'/ann_data'+str(model_num)+'/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast_512'



# checkpoint =  0 
# data_type = 1 
# test_set = 0 
# checkpoint_path ='/home/dihe/cudnn_file/recommender_shuqi/MIND_data/raw_data/'+str(exp)+'/ann_data'+str(model_num)+'/'
# raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast_512'

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
raw_data_dir = '/home/dihe/Projects/data/raw_data/'
# processed_data_dir = '/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast_512'



# if data_type == 0:
#     topN = 100
# else:
#     topN = 1000
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



dev_query_positive_id = {}
f=open('../../data/raw_data/qrels.dev.small.tsv','r')
for l in f:
    l = l.strip().split('\t')
    qid = int(l[0])
    # if qid in qids_to_relevant_passageids:
    #     pass
    # else:
    #     qids_to_relevant_passageids[qid] = []
    # qids_to_relevant_passageids[qid].append(int(l[2]))
    if qid not in dev_query_positive_id:
            dev_query_positive_id[qid] = {}
    dev_query_positive_id[qid][int(l[2])] = 1
    



data_type=1
test_set=0
# qidmap_path = processed_data_dir+"/qid2offset.pickle"
# pidmap_path = processed_data_dir+"/pid2offset.pickle"
if data_type == 0:
    if test_set == 1:
        query_path = raw_data_dir+"/msmarco-test2019-queries.tsv"
        passage_path = raw_data_dir+"/msmarco-doctest2019-top100"
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
    
# with open(qidmap_path, 'rb') as handle:
#     qidmap = pickle.load(handle)

# with open(pidmap_path, 'rb') as handle:
#     pidmap = pickle.load(handle)

# qset = set()
# with gzip.open(query_path, 'rt', encoding='utf-8') if query_path[-2:] == "gz" else open(query_path, 'rt', encoding='utf-8') as f:
#     tsvreader = csv.reader(f, delimiter="\t")
#     for [qid, query] in tsvreader:
#         qset.add(qid)

bm25 = collections.defaultdict(set)
with gzip.open(passage_path, 'rt', encoding='utf-8') if passage_path[-2:] == "gz" else open(passage_path, 'rt', encoding='utf-8') as f:
    for line in tqdm(f):
        if data_type == 0:
            [qid, Q0, pid, rank, score, runstring] = line.split(' ')
            pid = pid[1:]
        else:
            [qid, pid, query, passage] = line.split("\t")
            #print('???',qid)
        # if qid in qset and int(qid) in qidmap:
        #     bm25[qidmap[int(qid)]].add(pidmap[int(pid)])
        # else:
        #     print('???',qid,qid in qset)
        bm25[int(qid)].add(int(pid))


# bm25 = collections.defaultdict(set)
# f=open('../../data/raw_data/qrels.dev.small.tsv','r')
# for l in f:
#     l = l.strip().split('\t')
#     qid = int(l[0])
#     # if qid in qids_to_relevant_passageids:
#     #     pass
#     # else:
#     #     qids_to_relevant_passageids[qid] = []
#     # qids_to_relevant_passageids[qid].append(int(l[2]))
#     # if qid not in dev_query_positive_id:
#     #     dev_query_positive_id[qid] = {}
#     # dev_query_positive_id[qid][int(l[2])] = 1
#     bm25[int(qid)].add(int(l[2]))

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

def EvalDevQuery(query_embedding2id, passage_embedding2id, dev_query_positive_id, I_nearest_neighbor,topN):
    prediction = {} #[qid][docid] = docscore, here we use -rank as score, so the higher the rank (1 > 2), the higher the score (-1 > -2)

    total = 0
    labeled = 0
    Atotal = 0
    Alabeled = 0
    qids_to_ranked_candidate_passages = {} 
    for query_idx in range(len(I_nearest_neighbor)): 
        seen_pid = set()
        query_id = query_embedding2id[query_idx]
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
            pred_pid = passage_embedding2id[idx]
            #print('???',idx)
            
            if not pred_pid in seen_pid:
                # this check handles multiple vector per document
                qids_to_ranked_candidate_passages[query_id][rank]=pred_pid
                Atotal += 1
                #print('!!!!',pred_pid,dev_query_positive_id[query_id])
                if pred_pid not in dev_query_positive_id[query_id]:
                    Alabeled += 1
                    #print('???')
                if rank < 10:
                    total += 1
                    if pred_pid not in dev_query_positive_id[query_id]:
                        labeled += 1
                rank += 1
                prediction[query_id][pred_pid] = -rank
                seen_pid.add(pred_pid)

    # use out of the box evaluation script
    evaluator = pytrec_eval.RelevanceEvaluator(
        convert_to_string_id(dev_query_positive_id), {'map_cut', 'ndcg_cut', 'recip_rank','recall'})

    eval_query_cnt = 0
    result = evaluator.evaluate(convert_to_string_id(prediction))
    
    qids_to_relevant_passageids = {}
    for qid in dev_query_positive_id:
        qid = int(qid)
        if qid in qids_to_relevant_passageids:
            pass
        else:
            qids_to_relevant_passageids[qid] = []
            for pid in dev_query_positive_id[qid]:
                if pid>0:
                    qids_to_relevant_passageids[qid].append(pid)
            
    # if data_type == 0:
    #     MaxMRRRank=100
    # else:
    MaxMRRRank=10


    ms_mrr = compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages,MaxMRRRank=MaxMRRRank)

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


checkpoint_path='../../exp/flow_data/'
# dev_query_embedding = []
# dev_query_embedding2id = []
passage_embedding = []
passage_embedding2id = []
for i in range(8):
    #try:
    # print('???',checkpoint_path + "dev_query_"+str(checkpoint)+"__emb_p__data_obj_"+str(i)+".pb")
    # with open(checkpoint_path + "dev_query_"+str(checkpoint)+"__emb_p__data_obj_"+str(i)+".pb", 'rb') as handle:
    #     dev_query_embedding.append(pickle.load(handle))
    #     print('ok1???')
    # with open(checkpoint_path + "dev_query_"+str(checkpoint)+"__embid_p__data_obj_"+str(i)+".pb", 'rb') as handle:
    #     dev_query_embedding2id.append(pickle.load(handle))
    #     print('ok???',2)
    with open(checkpoint_path + "doc_emb"+str(i), 'rb') as handle:
        passage_embedding.append(np.array(pickle.load(handle),dtype="float32"))
        print('ok???',3)
    with open(checkpoint_path + "doc_id"+str(i), 'rb') as handle:
        passage_embedding2id.append(pickle.load(handle))
        print('ok???',4)
    # except:
    #     break
# if (not dev_query_embedding) or (not dev_query_embedding2id) or (not passage_embedding) or not (passage_embedding2id):
#     print("No data found for checkpoint: ",checkpoint)

# dev_query_embedding = np.concatenate(dev_query_embedding, axis=0)
# dev_query_embedding2id = np.concatenate(dev_query_embedding2id, axis=0)
passage_embedding = np.concatenate(passage_embedding, axis=0)

passage_embedding2id = np.concatenate(passage_embedding2id, axis=0)
dev_query_embedding=np.array(pickle.load(open('../../exp/flow_data/query_emb','rb')),dtype="float32")
dev_query_embedding2id=pickle.load(open('../../exp/flow_data/query_id','rb'))

print('???',type(passage_embedding),passage_embedding.shape,passage_embedding2id.shape,dev_query_embedding.shape,dev_query_embedding2id.shape)

# passage_embedding=pickle.load(open('../../data/dlow_data/passage_emb','rb'))
# passage_embedding2id=pickle.load(open('../../data/dlow_data/passage_emb_id','rb'))




#reranking

pidmap = collections.defaultdict(list)
for i in range(len(passage_embedding2id)):
    pidmap[passage_embedding2id[i]].append(i)  # abs pos(key) to rele pos(val)
    
rerank_data = {}
all_dev_I = []
for i,qid in enumerate(dev_query_embedding2id):
    p_set = []
    p_set_map = {}
    if qid not in bm25:
        print(qid,"not in bm25")
    else:
        count = 0
        for k,pid in enumerate(bm25[qid]):
            if pid in pidmap:
                for val in pidmap[pid]:
                    p_set.append(passage_embedding[val])
                    p_set_map[count] = val # new rele pos(key) to old rele pos(val)
                    count += 1
            else:
                print(pid,"not in passages")
    dim = passage_embedding.shape[1]
    faiss.omp_set_num_threads(16)
    cpu_index = faiss.IndexFlatIP(dim)
    p_set =  np.asarray(p_set)
    cpu_index.add(p_set)    
    _, dev_I = cpu_index.search(dev_query_embedding[i:i+1], len(p_set))
    for j in range(len(dev_I[0])):
        dev_I[0][j] = p_set_map[dev_I[0][j]]
    all_dev_I.append(dev_I[0])
result = EvalDevQuery(dev_query_embedding2id, passage_embedding2id, dev_query_positive_id, all_dev_I, topN)
final_ndcg, eval_query_cnt, final_Map, final_mrr, final_recall, hole_rate, ms_mrr, Ahole_rate, metrics, prediction = result
# print("Reranking Results for checkpoint "+str(checkpoint))
print("Reranking NDCG@10:" + str(final_ndcg))
print("Reranking map@10:" + str(final_Map))
print("Reranking pytrec_mrr:" + str(final_mrr))
print("Reranking recall@"+str(topN)+":" + str(final_recall))
print("Reranking hole rate@10:" + str(hole_rate))
print("Reranking hole rate:" + str(Ahole_rate))
print("Reranking ms_mrr:" + str(ms_mrr))


#full ranking
# dim = passage_embedding.shape[1]
# faiss.omp_set_num_threads(16)
# cpu_index = faiss.IndexFlatIP(dim)
# cpu_index.add(passage_embedding)    
# _, dev_I = cpu_index.search(dev_query_embedding, topN)
# result = EvalDevQuery(dev_query_embedding2id, passage_embedding2id, dev_query_positive_id, dev_I, topN)
# final_ndcg, eval_query_cnt, final_Map, final_mrr, final_recall, hole_rate, ms_mrr, Ahole_rate, metrics, prediction = result
# # print("Results for checkpoint "+str(checkpoint))
# print("NDCG@10:" + str(final_ndcg))
# print("map@10:" + str(final_Map))
# print("pytrec_mrr:" + str(final_mrr))
# print("recall@"+str(topN)+":" + str(final_recall))
# print("hole rate@10:" + str(hole_rate))
# print("hole rate:" + str(Ahole_rate))
# print("ms_mrr:" + str(ms_mrr))