import argparse
import os
import ruamel_yaml as yaml
import language_evaluation
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_caption_mplug import VSD
from models.vit import interpolate_pos_embed, resize_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset.utils import save_result
from dataset import create_dataset_all_HP, create_sampler, create_loader, coco_collate_fn_HP

from scheduler import create_scheduler
from optim import create_optimizer, create_two_optimizer

import sys


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config, do_amp=False,
          do_two_optim=False, do_accum=False, accum_steps=1, val_loader_1 = None, val_loader_2 = None, val_loader_3 = None,
              skip_global_step_zero = False, best_results_1 = None, best_results_2 = None, best_results_3 = None,
              best_results_epoch_1 = None, best_results_epoch_2 = None, best_results_epoch_3 = None):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if do_two_optim:
        metric_logger.add_meter('lr1', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
        metric_logger.add_meter('lr2', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    else:
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size
    for i, (image, HP_label) in enumerate(tqdm(data_loader, desc="Training")):
        image = image.to(device, non_blocking=True)
        
        if epoch > 0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1, i / len(data_loader))

        loss = model.forward_HP(image, torch.tensor(HP_label).to(image.device), train=True)
        if accum_steps > 1:
            loss = loss / accum_steps

        if do_amp:
            from apex import amp
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                # logger.info('scaled loss: {}'.format(str(scaled_loss)))
                scaled_loss.backward()
        else:
            loss.backward()
        if (i + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        metric_logger.update(loss=loss.item())

        if do_two_optim:
            metric_logger.update(lr1=optimizer.param_groups[0]["lr"])
            metric_logger.update(lr2=optimizer.param_groups[2]["lr"])
        else:
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)
        
        del image, loss 

            # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 50
    
    total_loss = 0
    num_data = 0
    result = []
    HP_pred_ = []
    HP_true = []
    
    answer_input = None
    for n, (image, HP_label) in enumerate(tqdm(data_loader, desc="Evaluation")):
        image = image.to(device,non_blocking=True) 
        
        logits, loss = model.forward_HP(image, torch.tensor(HP_label).to(image.device), train=False)
        
        logits = logits.detach().cpu().numpy()
        y_pred = np.argmax(logits, axis = 1)
        HP_pred_.extend(y_pred)
        HP_true.extend(HP_label)
        
        total_loss += loss
        num_data += image.size(0)
        
    return result, total_loss/num_data, torch.sum(torch.tensor(HP_pred_) == torch.tensor(HP_true))/len(HP_true)
                

@torch.no_grad()
def evaluate(model, data_loader, dataset, tokenizer, device, config):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 50
    predicts = []
    answers = []
    answer_input = None
    for n, (image, caption, image_ids, gold_caption) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):        
        image = image.to(device,non_blocking=True)             
        caption = [each+config['eos'] for each in caption]
        question_input = [config['bos']]*len(caption)
        caption = tokenizer(caption, padding='longest', truncation=True, max_length=args.max_input_length, return_tensors="pt").to(device)
        question_input = tokenizer(question_input, padding='longest', truncation=True, max_length=args.max_input_length, return_tensors="pt").to(device)

        for i in range(len(gold_caption)):
            predicts.append(gold_caption[i][0])
            answers.append(gold_caption[i])
        #{'Bleu_1': 0.9999999999863945, 'Bleu_2': 0.9999999999859791, 'Bleu_3': 0.9999999999854866, 'Bleu_4': 0.999999999984889, 'METEOR': 1.0, 'ROUGE_L': 1.0, 'CIDEr': 2.7246232035629268, 'SPICE': 0.40389416048620613}
        result = cal_metric(predicts, answers)
        metric_logger.meters['Bleu_1'].update(result["Bleu_1"], n=image.size(0))
        metric_logger.meters['Bleu_2'].update(result["Bleu_1"], n=image.size(0))
        metric_logger.meters['Bleu_3'].update(result["Bleu_1"], n=image.size(0))
        metric_logger.meters['Bleu_4'].update(result["Bleu_1"], n=image.size(0))
        metric_logger.meters['Bleu_1'].update(result["Bleu_1"], n=image.size(0))

    # gather the stats from all processes
    torch.cuda.empty_cache()
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

def cal_metric(result_file):
    result_list = json.load(open(result_file, "r"))
    predicts = []
    answers = []
    used_img_id = []
    for each in result_list:
        if each['question_id'] not in used_img_id:
            predicts.append(each["pred_caption"])
            answers.append(each["gold_caption"])
            used_img_id.append(each['question_id'])
        
    print('len(predicts) in cal_metric: ', len(predicts))
    
    evaluator = language_evaluation.CocoEvaluator(verbose=False)
    results = evaluator.run_evaluation(predicts, answers)
    print('='*100)
    print (len(result_list), results)
    print('='*100)
    return results

def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    print('seed: ', args.seed)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    #### Dataset ####
    print("Creating vqa datasets")
    datasets = create_dataset_all_HP('coco', config)
    
    for i in range(len(datasets)):
        print(f'datasets {i} length: ', len(datasets[i]))
        
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [True, False, False], num_tasks, global_rank)         
    else:
        samplers = [None, None, None, None, None, None, None, None, None, None]

    train_loader, val_loader_All \
    = create_loader(datasets,samplers, batch_size=[config['batch_size_train'],config['batch_size_test']],
                    num_workers=[8,8], 
                    is_trains=[True, False], 
                    collate_fns=[coco_collate_fn_HP, coco_collate_fn_HP]) 

    val_loaders = [val_loader_All]
    
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model ####
    print("Creating model")
    model = VSD(config=config, tokenizer=tokenizer)
    model = model.to(device)

    if not args.do_two_optim:
        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_optimizer(arg_opt, model)
    else:
        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_two_optimizer(arg_opt, model)

    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    if args.do_amp:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')

#         print('='*100)
#         print('keys in the checkpoint')
        for key in checkpoint.keys():
            print(key)
        
        try:
            state_dict = checkpoint['model']
        except:
            state_dict = checkpoint['module']

        # reshape positional embedding to accomodate for image resolution change
        if config["clip_name"] == "ViT-B-16":
            num_patches = int(config["image_res"] * config["image_res"]/(16*16))
        elif config["clip_name"] == "ViT-L-14":
            num_patches = int(config["image_res"] * config["image_res"]/(14*14))
        pos_embed = nn.Parameter(torch.zeros(num_patches + 1, 768).float())

        pos_embed = resize_pos_embed(state_dict['visual_encoder.visual.positional_embedding'].unsqueeze(0),
                                                   pos_embed.unsqueeze(0))
        state_dict['visual_encoder.visual.positional_embedding'] = pos_embed

        if not args.evaluate:
            for key in list(state_dict.keys()):
                if ('fusion' in key or 'bert' in key) and 'decode' not in key:
                    encoder_key = key.replace('fusion.', '').replace('bert.', '')
                    state_dict[encoder_key] = state_dict[key]
                    del state_dict[key]

#         print('='*77)
#         print('Parameters in the model: ')
#         for key in model.state_dict().keys():
#             print(key+ ' ==> ', model.state_dict()[key].size())
        
        model_state_dict = model.state_dict().copy()
        state_dict_keys = state_dict.keys()
        for key in state_dict_keys:
            if 'HP' in key:
                state_dict[key] = model_state_dict[key]

        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)
            
    model_without_ddp = model
    if args.distributed:
        #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        import apex
        model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
        model_without_ddp = model.module

    print('-'*77)
    print("Start training")
    best_results = [{'Bleu_1': 0.0, 
                   'Bleu_2': 0.0, 
                   'Bleu_3': 0.0, 
                   'Bleu_4': 0.0, 
                   'METEOR': 0.0, 
                   'ROUGE_L': 0.0, 
                   'CIDEr': 0.0, 
                   'SPICE': 0.0, } for _ in range(len(val_loaders))]
    best_results_epoch = [{'Bleu_1': 0.0, 
                   'Bleu_2': 0.0, 
                   'Bleu_3': 0.0, 
                   'Bleu_4': 0.0, 
                   'METEOR': 0.0, 
                   'ROUGE_L': 0.0, 
                   'CIDEr': 0.0, 
                   'SPICE': 0.0, } for _ in range(len(val_loaders))]


#     torch.save({
#                 'model': model_without_ddp.state_dict(),
#                 'optimizer': optimizer.state_dict(),
#                 'lr_scheduler': lr_scheduler.state_dict(),
#                 'config': config,
#             }, os.path.join('./', 'mPLUG_caption_base.pth'))


    start_time = time.time()

#     result_file = save_result(vqa_result, args.result_dir, 'vqa_result_epoch_-1')
#     if utils.is_main_process():
#         result = cal_metric(result_file)
#     dist.barrier()
    if 'epoch' in args.checkpoint:
        print('-'*77)
        print('Continue training from the provided checkpoint.')
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model'], strict = True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        
    for epoch in range(start_epoch, max_epoch):
        if epoch > 100000:
            torch.save({
                'epoch': epoch,
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
            }, os.path.join('./saved_checkpoints/', f'Recycle_epoch_{epoch}.pth'))
        
        if epoch > 0:
            lr_scheduler.step(epoch + warmup_steps)
            
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
                
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler,
                                         config, do_amp=args.do_amp, do_two_optim=args.do_two_optim, accum_steps=args.accum_steps)
            
            vqa_result, eval_loss, eval_acc = evaluation(model, val_loader_All, tokenizer, device, config)
            print('='*77)
            print(f'epoch {epoch}, eval_loss: {eval_loss: .3f}, eval_acc: {eval_acc: .3f}')
            if epoch == 8:
                torch.save({
                'model': model_without_ddp.state_dict()
                }, os.path.join('./', f'HP_epoch_{epoch}.pth'))
                sys.exit()
            

        if args.evaluate:
            break
                        
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('-'*77)
    print('Training time {}'.format(total_time_str))


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/VQA.yaml')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--output_dir', default='output/vqa')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--text_decoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--min_length', default=1, type=int)
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--max_length', default=10, type=int)
    parser.add_argument('--max_input_length', default=25, type=int)
    parser.add_argument('--beam_size', default=5, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--do_two_optim', action='store_true')
    parser.add_argument('--add_object', action='store_true')
    parser.add_argument('--do_amp', action='store_true')
    parser.add_argument('--no_init_decocde', action='store_true')
    parser.add_argument('--do_accum', action='store_true')
    parser.add_argument('--accum_steps', default=4, type=int)
    parser.add_argument('--eval_start_epoch', default=4, type=int)
    parser.add_argument('--save_results_dir', default='', type =str)
    
    
    # By Ikhyun Cho
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    Path(args.save_results_dir).mkdir(parents=True, exist_ok=True)
    config["min_length"] = args.min_length
    config["max_length"] = args.max_length
    config["add_object"] = args.add_object
    config["beam_size"] = args.beam_size
    #config['optimizer']['lr'] = args.lr
    #config['schedular']['lr'] = args.lr
    config['text_encoder'] = args.text_encoder
    config['text_decoder'] = args.text_decoder


    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)
