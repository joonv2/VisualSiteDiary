from functools import partial
from models.vit import VisionTransformer
from models.modeling_mplug import BertConfig, BertModel, BertPrefixModel, FusionModel
from models.visual_transformers import initialize_clip
from models.predictor import TextGenerator

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import numpy as np

class PR(nn.Module):
    """PR layer implementation
    
    Args:
        num_clusters : int
            The number of pseudo regions
        dim : int
            Dimension of pseudo regions
        alpha : float
            Parameter of initialization. Larger value is harder assignment.
        normalize_input : bool
            If true, pseudo regions-wise L2 normalization is applied to input.
    """
    def __init__(self, num_regions=64, dim=768, normalize_input=True): #alpha=100.0, NORM= TRUE
        super().__init__()
        self.num_regions = num_regions
        self.dim = dim
        #self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_regions, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_regions, dim))
        self.init_weights()
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def forward(self, grids):
                   
        N, C = grids.shape[0], grids.shape[-1]

        grids = grids.view(N, 577, 768, -1).permute(0,2,1,3).contiguous()
        #print('grids,', grids.shape) #grids, torch.Size([64, 768, 577, 1])

        if self.normalize_input:
            grids = F.normalize(grids, p=2, dim=1)  # across descriptor dim

        soft_assign = self.conv(grids).view(N, self.num_regions, -1)
        #print('soft_assing,', soft_assign.shape) #soft_assing, torch.Size([64, 4, 577])
        soft_assign = F.softmax(soft_assign, dim=1)
        #print('soft_assing,', soft_assign.shape)
        x_flatten = grids.view(N, C, -1)
        
        residual = x_flatten.expand(self.num_regions, -1, -1, -1).permute(1, 0, 2, 3).contiguous() - \
            self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).contiguous().unsqueeze(0)
        #print('residual,', residual.shape)
        residual *= soft_assign.unsqueeze(2)
        p = residual.sum(dim=-1)
        #print('residual,', residual.shape)
        p = F.normalize(p, p=2, dim=2)  # intra-normalization
        p = p.view(grids.size(0), -1)
        p = F.normalize(p, p=2, dim=1)  # L2 normalize
        #print("p shape. ", p.shape) #p shape.  torch.Size([64, 3072])
        return p
    
class VSD(nn.Module):
    def __init__(self,                 
                 tokenizer = None,
                 config = None,
                 use_PR = False,
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer 
        self.module_setting(config)
        self.visual_encoder, _ = initialize_clip(config)
        #self.text_encoder = BertModel.from_pretrained(config['text_encoder'], config=self.config_encoder, add_pooling_layer=False)  
        #self.fusion_encoder = FusionModel.from_pretrained(config['text_encoder'], config=self.config_fusion, add_pooling_layer=False)  
        self.text_decoder = BertPrefixModel.from_pretrained(config['text_decoder'], config=self.config_decoder)    
        self.beam_generator = TextGenerator(config, self.text_decoder) 
        
        ## For HP
        self.HP_layer = nn.Linear(768,4)
        self.tanh = nn.Tanh()
        
        ## For PR
        self.use_PR = use_PR
        if use_PR:
            print('Using Pseudo Region Layers')
            self.PR = PR(4, dim=768)
            
        
    def forward(self, image, question, answer=None, train=True, out_size=5, scst=False):
        if(scst):
            return self.beam_search(image, question, answer, train=True,out_size=out_size)
        image = image.to(dtype=next(self.parameters()).dtype)
        
        # image.size(): torch.Size([64, 3, 384, 384]) => batch_size = 64, image_resolution = 384
        
        image_embeds = self.visual_encoder.visual(image, skip_last_layer=True, use_checkpoint=self.use_checkpoint)
        
        # image_embeds.size() : torch.Size([64, 577, 768]) => batch_size = 64, 
        # Where does 577 come from?? 1 + 24^2. 따라서 아마 image patch가 24x24들어 있는 듯?
        # 즉, 1 patch = 32x32 => 1 image (384x384) = 24x24 patches. 1 for [CLS].
        
        if self.large:
            image_embeds = self.dropout(self.visn_layer_norm(self.visn_fc(image_embeds)))
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        
        ####
        #For Pseudo Region 
        if self.use_PR:
            bs, __, vis_dim = image_embeds.size()
            #pseudo_region = self.SP(image_embeds)
            #print("pseudo_region shape, ", pseudo_region.shape) #pseudo_region shape,  torch.Size([64, 3072])
            pseudo_region = self.PR(image_embeds).view(bs, 4, vis_dim)
            #print("pseudo_region new shape, ", pseudo_region.shape) #pseudo_region new shape,  torch.Size([64, 4, 768])
            #####
            
            region_atts = torch.ones(pseudo_region.size()[:-1],dtype=torch.long).to(image.device)
            image_embeds, image_atts = torch.cat([image_embeds, pseudo_region],dim=1), torch.cat([image_atts, region_atts], dim=-1)
        
        if train:               
            answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)
            # answer.input_ids.size() : torch.Size([64,25]) 가끔 torch.Size([64,24])?
            # answer_targets.size() : torch.Size([64,25])
            # answer_targets는 위 코드를 보면 알겠지만 0 (pad_token_id) 인 것들을 -100으로 바꾼 것.
            # 실제 예시:
            # input_ids[0] -> torch.tensor([  101,  1996,  2067, 14490,  7170,  2121, 15199,  1996,  2455,  1998, ....])
            # tokenizer.convert_ids_to_tokens(input_ids[0]) = ['[CLS]', 'the', 'back', '##hoe', 'load', '##er', 'excavated',
            # 'the', 'land', 'and', 'is', 'traveling', 'loaded', '[SEP]', '[PAD]' , ...]
            # answer.attention_mask[0] = answer.input_ids 실제 토큰까지 1 그 뒤 padding 부분은 전부 0. 
            
            answer_output = self.text_decoder(answer.input_ids, 
                                                  attention_mask = answer.attention_mask, 
                                                  encoder_hidden_states = image_embeds,
                                                  encoder_attention_mask = image_atts,                  
                                                  labels = answer_targets,
                                                  return_dict = True,   
                                                  reduction = 'none',
                                                 )
            
            # answer_output[0] = loss=tensor(4.0908, device='cuda:0', grad_fn=<NllLossBackward0>)
            # answer_output[1] = logits= tensor with size torch.Size([64, 25, 30522])
            #   * 30522 seems to be the vocab size. 64 is the batch size. 25 is the maximum caption length
            #   * mPLUG/output/cicd_caption_base_1e-5/config.yaml 보면 나와 있음.
            
            
            loss = answer_output.loss         

            return loss
            

        else: 
            topk_ids, topk_probs = self.generation(image_embeds, image_atts)
            # len(topk_ids) = 64
            # len(topk_ids[0]) = 1 
            # topk_ids[0][0].size() => depends on the data. torch.Size([10]), torch.Size([13]), torch.Size([12]) 등등
            # 예시) topk_ids[0][0] = tensor([  101,  1996,  4654,  3540, 22879, ... ,   102], device='cuda:0')
            return topk_ids, topk_probs
        
    def forward_HP(self, image, HP_label = None, train=True, out_size=5, scst=False):
        if(scst):
            return self.beam_search(image, question, answer, train=True,out_size=out_size)
        image = image.to(dtype=next(self.parameters()).dtype)
        
        # image.size(): torch.Size([64, 3, 384, 384]) => batch_size = 64, image_resolution = 384
        image_embeds = self.visual_encoder.visual(image, skip_last_layer=True, use_checkpoint=self.use_checkpoint)
        
        logits = self.HP_layer(self.tanh(image_embeds[:,0]))
        
        if train:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 4), HP_label.view(-1))
            return loss
        else:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 4), HP_label.view(-1))
            return logits, loss
            
    def module_setting(self, config):
        self.config_encoder = BertConfig.from_json_file(config['bert_config'])   
        self.config_encoder.num_hidden_layers = self.config_encoder.text_encoder_layers
        self.config_fusion = BertConfig.from_json_file(config['bert_config'])   
        self.config_decoder = BertConfig.from_json_file(config['bert_config'])
        self.config_decoder.add_cross_attention = True
        self.config_decoder.num_hidden_layers = self.config_decoder.text_decode_layers
        self.large = False
        if self.config_encoder.hidden_size != config['vision_width']:
            self.visn_fc = nn.Linear(config['vision_width'], self.config_encoder.hidden_size)
            self.visn_layer_norm = nn.LayerNorm(self.config_encoder.hidden_size, eps=1e-12)
            self.dropout = nn.Dropout(self.config_encoder.hidden_dropout_prob)
            self.large = True
        self.use_checkpoint = config["use_checkpoint"] if "use_checkpoint" in config else True
        print ("use_checkpoint: ", self.use_checkpoint)

    def beam_search(self, image, question, answer=None, train=True, out_size=5):
        image_embeds = self.visual_encoder.visual(image, skip_last_layer=True)
        if self.large:
            image_embeds = self.dropout(self.visn_layer_norm(self.visn_fc(image_embeds)))
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        topk_ids, topk_probs = self.generation(image_embeds, image_atts, out_size=out_size) 

        return topk_ids, topk_probs
    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
    def generation(self, question_states, question_atts, out_size=1):
        encoder_inputs = [question_states, question_atts]
        topk_ids,topk_probs = self.beam_generator.translate_batch_scst(encoder_inputs,out_size=out_size)  
        return topk_ids, topk_probs

    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):
        
        num_ques = question_states.size(0)
        start_ids = answer_ids[0,0].repeat(num_ques,1) # bos token
        
        start_output = self.text_decoder(start_ids, 
                                         encoder_hidden_states = question_states,
                                         encoder_attention_mask = question_atts,                                      
                                         return_dict = True,
                                         reduction = 'none')              
        logits = start_output.logits[:,0,:] # first token's logit
        
        # topk_probs: top-k probability 
        # topk_ids: [num_question, k]        
        answer_first_token = answer_ids[:,1]
        prob_first_token = F.softmax(logits,dim=1).index_select(dim=1, index=answer_first_token) 
        topk_probs, topk_ids = prob_first_token.topk(k,dim=1) 
        
        # answer input: [num_question*k, answer_len]                 
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids,dim=0)  
        input_atts = torch.cat(input_atts,dim=0)  

        targets_ids = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)

        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)
        
        output = self.text_decoder(input_ids, 
                                   attention_mask = input_atts, 
                                   encoder_hidden_states = question_states,
                                   encoder_attention_mask = question_atts,     
                                   labels = targets_ids,
                                   return_dict = True, 
                                   reduction = 'none')                 

        answer_loss = output.loss 
        answer_loss = answer_loss.view(input_ids.size(0),-1)
        
        # topk_prob: first token probability
        topk_probs = topk_probs.view(-1,1)
        log_probs = torch.cat([topk_probs.log(), -answer_loss],dim=1)

        # re-calculate log probabilities for the answer sequences using chain rule
        log_probs_sum = log_probs.sum(1)
        log_probs_sum = log_probs_sum.view(num_ques,k)

        topk_probs = F.softmax(log_probs_sum, dim=-1)
        # get top-k after re-ranking
        topk_probs, rerank_id = topk_probs.topk(k,dim=1) 
        topk_ids = torch.gather(topk_ids, 1, rerank_id)    

        return topk_ids, topk_probs
    
def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))    
