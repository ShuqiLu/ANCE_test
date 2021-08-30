import sys
sys.path += ['../']
import torch
from torch import nn
from transformers import (
    RobertaConfig,
    RobertaModel,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    BertModel,
    BertTokenizer,
    BertConfig,
    BertForSequenceClassification, 
    ElectraTokenizer, 
    ElectraModel,
    ElectraForSequenceClassification,
    ElectraConfig
)
import torch.nn.functional as F
from data.process_fn import triple_process_fn, triple2dual_process_fn
from torch import Tensor as T
from typing import Tuple
# from fairseq.modules import (
#     LayerNorm,
#     MultiheadAttention,
#     PositionalEmbedding,
#     TransformerSentenceEncoderLayer,
# )
from fairseq.modules import (
    TransformerSentenceEncoder,
)
from transformers import ElectraTokenizer, ElectraModel
from transformers import AutoTokenizer, AutoModel

from models.SEED_Encoder import SEEDEncoderConfig, SEEDTokenizer, SEEDEncoderForSequenceClassification


class EmbeddingMixin:
    """
    Mixin for common functions in most embedding models. Each model should define its own bert-like backbone and forward.
    We inherit from RobertaModel to use from_pretrained 
    """
    def __init__(self, model_argobj):
        if model_argobj is None:
            self.use_mean = False
        else:
            self.use_mean = model_argobj.use_mean
        print("Using mean:", self.use_mean)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def masked_mean(self, t, mask):
        s = torch.sum(t * mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d

    def masked_mean_or_first(self, emb_all, mask):
        # emb_all is a tuple from bert - sequence output, pooler
        if isinstance(emb_all, tuple):
            if self.use_mean:
                return self.masked_mean(emb_all[0], mask)
            else:
                return emb_all[0][:, 0]
        else:
            #print('!!!',emb_all.shape)
            if self.use_mean:
                return self.masked_mean(emb_all, mask)
            else:
                #print('??? should be the first')
                return emb_all[:, 0]

    def query_emb(self, input_ids, attention_mask):
        raise NotImplementedError("Please Implement this method")

    def body_emb(self, input_ids, attention_mask):
        raise NotImplementedError("Please Implement this method")


class NLL(EmbeddingMixin):
    def forward(
            self,
            query_ids,
            attention_mask_q,
            input_ids_a=None,
            attention_mask_a=None,
            input_ids_b=None,
            attention_mask_b=None,
            is_query=True):
        if input_ids_b is None and is_query:
            return self.query_emb(query_ids, attention_mask_q)
        elif input_ids_b is None:
            return self.body_emb(query_ids, attention_mask_q)

        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        b_embs = self.body_emb(input_ids_b, attention_mask_b)

        #print('???',q_embs.shape,a_embs.shape)

        logit_matrix = torch.cat([(q_embs * a_embs).sum(-1).unsqueeze(1),
                                  (q_embs * b_embs).sum(-1).unsqueeze(1)], dim=1)  # [B, 2]
        #print('???',torch.cosine_similarity(q_embs,a_embs),torch.cosine_similarity(q_embs,a_embs).shape)



        # logit_matrix = torch.cat([torch.cosine_similarity(q_embs,a_embs).unsqueeze(1), 
        #                           torch.cosine_similarity(q_embs,b_embs).unsqueeze(1)], dim=1)

        #print('???',logit_matrix.shape,logit_matrix)
        lsm = F.log_softmax(logit_matrix, dim=1)
        # #print('???',lsm)
        #assert 1==0
        loss = -1.0 * lsm[:, 0]

        acc=lsm[:,0]>lsm[:,1]

        # score_a=(q_embs * a_embs).sum(-1)#.unsqueeze(1)
        # score_b=(q_embs * b_embs).sum(-1)#.unsqueeze(1)
        # target=torch.ones(score_a.shape).type_as(score_a)

        # loss=torch.nn.MarginRankingLoss(margin=1.0)(score_a,score_b,target)
        # return (loss.mean(),)

        #q_embs_norm_avg=torch.sum(torch.norm(q_embs,dim=1))/q_embs.shape[0]
        # a_embs_norm_avg=torch.sum(torch.norm(a_embs,dim=1))/a_embs.shape[0]
        # b_embs_norm_avg=torch.sum(torch.norm(b_embs,dim=1))/b_embs.shape[0]
        # cls_norm=(a_embs_norm_avg+b_embs_norm_avg)/2
        # # print('???',torch.norm(q_embs,dim=1),torch.norm(a_embs,dim=1),torch.norm(b_embs,dim=1))
        # # #print("???",query_ids,input_ids_a,input_ids_b)
        # return (loss.mean(),cls_norm)

        
        # cls_sim=torch.sum(torch.cosine_similarity(a_embs, b_embs, dim=1))/b_embs.shape[0]
        #cls_sim=torch.sum(torch.cosine_similarity(a_embs, b_embs, dim=1))/b_embs.shape[0]
        #print("???",F.mse_loss(a_embs, b_embs))
        #cls_sim=torch.sum(F.mse_loss(a_embs, b_embs,reduction='none').sum(1))/b_embs.shape[0]
        # cls_simb=torch.sum(torch.cosine_similarity(q_embs, b_embs, dim=1))/b_embs.shape[0]
        # cls_sima=torch.sum(torch.cosine_similarity(q_embs, a_embs, dim=1))/b_embs.shape[0]
        # cls_sima=(q_embs * a_embs).sum(-1)
        # cls_simb=(q_embs * b_embs).sum(-1)
        #cls_sim= torch.sum((b_embs * a_embs).sum(-1))/b_embs.shape[0]
        # print('cls_sima',cls_sima,'cls_simb',cls_simb)
        return (loss.mean(),acc.float().mean())

class RobertaDot_NLL_LN(NLL, RobertaForSequenceClassification):
    """None
    Compress embedding to 200d, then computes NLL loss.
    """

    def __init__(self, config, model_argobj=None):
        NLL.__init__(self, model_argobj)
        RobertaForSequenceClassification.__init__(self, config)
        self.embeddingHead = nn.Linear(config.hidden_size, 768)
        self.norm = nn.LayerNorm(768)
        self.apply(self._init_weights)

    def query_emb(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.norm(self.embeddingHead(full_emb))
        return query1

    def body_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)


class RobertaDot_NLL_LN_fairseq_fast(NLL,nn.Module):
    """None
    Compress embedding to 200d, then computes NLL loss.
    """
    def __init__(self, config, model_argobj=None):
        nn.Module.__init__(self)
        NLL.__init__(self, model_argobj)
        
        self.encoder=TransformerSentenceEncoder(
                padding_idx=1,
                vocab_size=32769,
                num_encoder_layers=12,
                embedding_dim=768,
                ffn_embedding_dim=3072,
                num_attention_heads=12,
                dropout=0.1,
                attention_dropout=0.1,
                activation_dropout=0.0,
                layerdrop=0.0,
                max_seq_len=512,
                num_segments=0,
                encoder_normalize_before=True,
                apply_bert_init=True,
                activation_fn="gelu",
                q_noise=0.0,
                qn_block_size=8,
        )
        self.embeddingHead = nn.Linear(config.hidden_size, 768)
        self.norm = nn.LayerNorm(768)
        self.apply(self._init_weights)

    def query_emb(self, input_ids, attention_mask):
        #print('???input_ids',input_ids.shape)
        outputs1, _ = self.encoder(input_ids)#[-1].transpose(0,1)
        #print('???',outputs1)
        outputs1=outputs1[-1].transpose(0,1)
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.norm(self.embeddingHead(full_emb))

        # query_norm=torch.norm(query1,dim=1).unsqueeze(-1)
        # query1=query1/query_norm
        return query1

    def body_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)

    #tikenid pad
    def from_pretrained(self, model_path):
        model_dict = self.state_dict()
        save_model=torch.load(model_path, map_location=lambda storage, loc: storage)
        #print(save_model['model'].keys())
        pretrained_dict= {}
        # print('???model_dict',model_dict.keys(),len(model_dict.keys()))
        # print('???save_model',save_model['model'].keys(),len(save_model['model'].keys()))
        if 'model' in save_model.keys():
            #save_model['model']
            for name in save_model['model']:
                if 'lm_head' not in name and 'encoder' in name and 'decode' not in name:
                    pretrained_dict['encoder'+name[24:]]=save_model['model'][name]
                # if  'lm_head' not in name and 'decode' not in name:
                #     if 'encoder' not in name:
                #         pretrained_dict[name]=save_model['model'][name]
                #     else:
                #         pretrained_dict['encoder'+name[24:]]=save_model['model'][name]
            
            # assert len(model_dict)-4==len(pretrained_dict)
            # for item in pretrained_dict.keys():
            #     if item not in model_dict:
            #         print('???',item)
            assert len(model_dict)-4==len(pretrained_dict), (len(model_dict),len(pretrained_dict),model_dict.keys(),pretrained_dict.keys())
        else:
            print('load finetuned checkpoints...')
            for name in save_model:
                pretrained_dict[name[7:]]=save_model[name]
            assert len(model_dict)==len(pretrained_dict)

        #print(model_dict.keys())
        print('load model.... ',len(model_dict),len(pretrained_dict))
        print(pretrained_dict.keys())
        
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        #pass


class RobertaDot_NLL_LN(NLL, SEEDEncoderForSequenceClassification):
    """None
    Compress embedding to 200d, then computes NLL loss.
    """
    def __init__(self, config, model_argobj=None):
        NLL.__init__(self, model_argobj)
        SEEDEncoderForSequenceClassification.__init__(self, config)
        self.embeddingHead = nn.Linear(config.encoder_embed_dim, 768)
        self.norm = nn.LayerNorm(768)
        self.apply(self._init_weights)

    def query_emb(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.norm(self.embeddingHead(full_emb))
        return query1

    def body_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)




# --------------------------------------------------
ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys()) 
        for conf in (
            RobertaConfig,
        ) if hasattr(conf,'pretrained_config_archive_map')
    ),
    (),
)


default_process_fn = triple_process_fn


class MSMarcoConfig:
    def __init__(self, name, model, process_fn=default_process_fn, use_mean=True, tokenizer_class=RobertaTokenizer, config_class=RobertaConfig):
        self.name = name
        self.process_fn = process_fn
        self.model_class = model
        self.use_mean = use_mean
        self.tokenizer_class = tokenizer_class
        self.config_class = config_class


configs = [
    MSMarcoConfig(name="rdot_nll",
                model=RobertaDot_NLL_LN,
                use_mean=False,
                ),
    
    MSMarcoConfig(name="rdot_nll_fairseq_fast",
                model=RobertaDot_NLL_LN_fairseq_fast,
                use_mean=False,
                #config_class=,
                ),
    MSMarcoConfig(name="rdot_nll_seed_encoder",
                model=RobertaDot_NLL_LN,
                use_mean=False,
                tokenizer_class=SEEDTokenizer,
                config_class=SEEDEncoderConfig,
                ),
    
]

MSMarcoConfigDict = {cfg.name: cfg for cfg in configs}
