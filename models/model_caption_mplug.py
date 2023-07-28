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
    Code taken from S2-Transformer https://github.com/zchoi/S2-Transformer
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

        if self.normalize_input:
            grids = F.normalize(grids, p=2, dim=1)  # across descriptor dim

        soft_assign = self.conv(grids).view(N, self.num_regions, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        x_flatten = grids.view(N, C, -1)
        
        residual = x_flatten.expand(self.num_regions, -1, -1, -1).permute(1, 0, 2, 3).contiguous() - \
            self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).contiguous().unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)
        p = residual.sum(dim=-1)
        p = F.normalize(p, p=2, dim=2)  # intra-normalization
        p = p.view(grids.size(0), -1)
        p = F.normalize(p, p=2, dim=1)  # L2 normalize
        return p
    
class MPLUG(nn.Module):
    def __init__(self,                 
                 tokenizer = None,
                 config = None,
                 use_PR = True,
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
        self.HP_layer_1 = nn.Linear(768,768)
        self.tanh = nn.Tanh()
        self.HP_layer_2 = nn.Linear(768,4)
        
        self.HP_layer_1.weight.data.normal_(mean=0.0, std=0.02)
#         if self.HP_layer_1.bias is not None:
#             self.HP_layer_1.bias.data.zero_()
                
        self.HP_layer_2.weight.data.normal_(mean=0.0, std=0.02)
#         if self.HP_layer_2.bias is not None:
#             self.HP_layer_2.bias.data.zero_()

        ## For PR
        self.use_PR = use_PR
        if use_PR:
            print('Using Pseudo Region Layers')
            self.PR = PR(4, dim=768)
            
        
    def forward(self, image, question, answer=None, train=True, out_size=5, scst=False):
        if(scst):
            return self.beam_search(image, question, answer, train=True,out_size=out_size)
        image = image.to(dtype=next(self.parameters()).dtype)

        image_embeds = self.visual_encoder.visual(image, skip_last_layer=True, use_checkpoint=self.use_checkpoint)

        if self.large:
            image_embeds = self.dropout(self.visn_layer_norm(self.visn_fc(image_embeds)))
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        
        ####
        #For Pseudo Region 
        if self.use_PR:
            bs, __, vis_dim = image_embeds.size()
            pseudo_region = self.PR(image_embeds).view(bs, 4, vis_dim)
            region_atts = torch.ones(pseudo_region.size()[:-1],dtype=torch.long).to(image.device)
            image_embeds, image_atts = torch.cat([image_embeds, pseudo_region],dim=1), torch.cat([image_atts, region_atts], dim=-1)
        
        if train:               
            answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)
            
            answer_output = self.text_decoder(answer.input_ids, 
                                                  attention_mask = answer.attention_mask, 
                                                  encoder_hidden_states = image_embeds,
                                                  encoder_attention_mask = image_atts,                  
                                                  labels = answer_targets,
                                                  return_dict = True,   
                                                  reduction = 'none',
                                                 )
            loss = answer_output.loss
            return loss
            

        else: 
            topk_ids, topk_probs = self.generation(image_embeds, image_atts)
            return topk_ids, topk_probs
        
    def forward_HP(self, image, HP_label = None, train=True, out_size=5, scst=False):
        if(scst):
            return self.beam_search(image, question, answer, train=True,out_size=out_size)
        image = image.to(dtype=next(self.parameters()).dtype)
        image_embeds = self.visual_encoder.visual(image, skip_last_layer=True, use_checkpoint=self.use_checkpoint)
        
        logits = self.HP_layer_2(self.tanh(self.HP_layer_1(image_embeds[:,0])))
        
        if train:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 4), HP_label.view(-1))
            return loss
        else:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 4), HP_label.view(-1))
            return logits, loss
            
        if self.large:
            image_embeds = self.dropout(self.visn_layer_norm(self.visn_fc(image_embeds)))
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        
        if train:               
            answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)
            
            answer_output = self.text_decoder(answer.input_ids, 
                                                  attention_mask = answer.attention_mask, 
                                                  encoder_hidden_states = image_embeds,
                                                  encoder_attention_mask = image_atts,                  
                                                  labels = answer_targets,
                                                  return_dict = True,   
                                                  reduction = 'none',
                                                 )

            loss = answer_output.loss
            return loss
            

        else: 
            topk_ids, topk_probs = self.generation(image_embeds, image_atts)
            return topk_ids, topk_probs
 

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
