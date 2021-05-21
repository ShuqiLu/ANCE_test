import pickle
ann_path='/home/dihe/Projects/data/raw_data/exp_01_05_09/ann_training_data_0'
processed_data_dir_query_origin='/home/dihe/Projects/data/raw_data/ann_data_roberta-base-fast-doc_512/'
with open(processed_data_dir_query_origin+"/qid2offset_train.pickle", 'rb') as handle:
    qidmap_origin = pickle.load(handle)


qidmap_origin_re={}
for item in qidmap_origin:
    assert qidmap_origin[item] not in qidmap_origin_re
    qidmap_origin_re[qidmap_origin[item]]=item


train_q_sample20={}
with open(ann_path, 'r') as f:
    ann_training_data = f.readlines()

file_dict={}
f2=open('../../data/raw_data/msmarco-doctrain-queries.tsv','r')
for line in f2:
	line_t=line.split('\t')
	file_dict[int(line_t[0])]=line


w=open('../../data/raw_data/msmarco-doctrain-queries-small2.tsv','w')
for line in ann_training_data:
    line_arr=line.strip().split('\t')
    qid = qidmap_origin_re[int(line_arr[0])]
    w.write(file_dict[qid])
w.close()

