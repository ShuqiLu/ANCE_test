from fairseq.modules import (
    TransformerSentenceEncoder,
)
class SEEDEncoder(nn.Module):
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

    def foward(self, input_ids, attention_mask=None):
        #print('???input_ids',input_ids.shape)
        outputs1, _ = self.encoder(input_ids)#[-1].transpose(0,1)
        #print('???',outputs1)
        outputs1=outputs1[-1].transpose(0,1)
        cls_output=outputs1[:,0]
        
        return cls_output

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
            #assert len(model_dict)-4==len(pretrained_dict), (len(model_dict),len(pretrained_dict),model_dict.keys(),pretrained_dict.keys())
            assert len(model_dict)==len(pretrained_dict)
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