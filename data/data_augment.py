import sys
import os
import torch
sys.path += ['../']
import gzip
import pickle
from utils.util import pad_input_ids, multi_file_process, numbered_byte_file_generator, EmbeddingCache
import csv
from model.models import MSMarcoConfigDict, ALL_MODELS
from torch.utils.data import DataLoader, Dataset, TensorDataset, IterableDataset, get_worker_info
import numpy as np
from os import listdir
from os.path import isfile, join
import argparse
import json
import csv

        
def find_dev_doc():
    
    
    '''
    dev_rel_path='../../raw_data/msmarco-docdev-qrels.tsv'
    dev_pos_d_dict={}
            
    with open(dev_rel_path,'r',encoding='utf-8') as f:
        tsvreader = csv.reader(f, delimiter=" ")
        for [topicid, _, docid, rel] in tsvreader:
            #topicid = int(topicid)
            #docid = int(docid[1:])
            dev_pos_d_dict[docid]=1
    '''

    '''
    dev_rel_path='../../data/raw_data/msmarco-docs.tsv'
    dev_pos_d_dict={}
            
    with open(dev_rel_path,'r',encoding='utf-8') as f:
        for line in f:
            line_arr = line.split('\t')
            p_id = line_arr[0]
            dev_pos_d_dict[p_id]=1   
    #但是数据放大了10倍

    
    all_rel_path='../../data/raw_data/msmarco-doctrain-qrels.tsv'
    all_pos_d_dict={}

    with open(all_rel_path,'r',encoding='utf-8') as f:
        tsvreader = csv.reader(f, delimiter=" ")
        for [topicid, _, docid, rel] in tsvreader:
            #topicid = int(topicid)
            #docid = int(docid[1:])
            all_pos_d_dict[docid]=1

    
    d_to_add={}
    for item in dev_pos_d_dict:
        if item not in all_pos_d_dict:
            d_to_add[item]=1

    print('dev doc: ',len(d_to_add),' add doc: ',len(all_pos_d_dict))
    #print(d_to_add)
    '''

    
    all_doc_path='../../data/raw_data/msmarco-docs.tsv'
    all_doc_dict={}
    with open(all_doc_path,'r', encoding='utf-8') as f:
        for line in f:
            d_id = line.split('\t')[0]
            #print(d_id)
            #assert 1==0
            #if d_id in d_to_add:
            all_doc_dict[d_id]=line
    print(len(all_doc_dict))
    d_to_add=all_doc_dict
    w=open('../../ContrastQG/exp_qg_all/corpus.jsonl','w', encoding="utf-8")
    for item in d_to_add:
        content=all_doc_dict[item]
        content=content.split('\t')
        data={"_id":item,"title":content[2],"text":content[3],"metadata":{}}
        w.write("{}\n".format(json.dumps(data)))
    w.close()



def get_new_file():
    origin_query=open('../../data/raw_data/msmarco-doctrain-queries.tsv','r')
    new_query=open('../../data/raw_data/exp_qg_all/queries.jsonl','r')
    query_w=open('../../data/raw_data/exp_qg_all/new_query.tsv','w')
    tsv_q_w = csv.writer(query_w, delimiter='\t')
    rel_w=open('../../data/raw_data/exp_qg_all/new_rel.tsv','w')
    tsv_r_w = csv.writer(rel_w, delimiter=' ')
    max_num=0
    for line in origin_query:
        line_arr = line.split('\t')
        qid = int(line_arr[0])
        if qid>max_num:
            max_num=qid
    max_num+=1

    for line in new_query:
        data=json.loads(line)
        tsv_q_w.writerow([str(max_num),data['text']])
        tsv_r_w.writerow([str(max_num),str(0),str(data['pos_doc_id']),str(1)])
        max_num+=1

    query_w.close()
    rel_w.close()




find_dev_doc()
#get_new_file()



    
            
        
    

    
