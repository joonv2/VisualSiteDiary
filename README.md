# VisualSiteDiary
An image captioning model built upon a pretrained ViT model (mPLUG) that provides human-readable captions to decipher daily progress and work activities from construction photologs
## Introduction
We present VisualSiteDiary that provides human-readable captions to decipher daily progress and work activities from construction photologs. To achieve high-quality descriptions from these photologs, our method: 1) incorporates pseudo region features, 2) utilizes high-level knowledge for pretraining the model, and 3) fine-tunes the model to consider different styles of captions accommodating different construction use-cases such as daily construction reporting. To validate the model and enable future research, we present VSD, a new comprehensive image captioning dataset that demonstrates various construction scenarios such as site work and super/sub-structure activities. Experiments show that \MethodName provides superior-quality captions compared to the state-of-the-art image captioning models on the VSD dataset. The dataset itself also offers more realistic yet challenging cases that need to be considered for real application on a project. We discuss how our method can be used for 1) daily construct reporting, and 2) image retrieval from existing photologs. Examples are also shared on how \MethodName enables variable style-caption generation from challenging construction images. 

<img src="VSD_framework.png" width="600"> 


## Pre-trained models and datasets

* Pre-trained models

 
For VQA and image captioning tasks, we do an additional continue pre-training on 4M image-text pairs based mplug.en.large to get mplug.en.large.v2.
 
 
|Model | Visual Backbone | Text Enc Layers | Fusion Layers | Text Dec Layers | #params | Download |
|------------------------|-------------------------------------------|------|------|------|------|-----|
|visualsitediary.base | [vit-b-16](https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/mPLUG/ViT-B-16.tar) | 6 | 6 | 12 | --M | [visualsitediary.base](https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/mPLUG/mplug_base.pth) |
|visualsitediary.compact | [vit-b-16](https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/mPLUG/ViT-B-16.tar) | 6 | 6 | 12 | --M | [visualsitediary.compact](https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/mPLUG/mplug_large.pth) |
|visualsitediary.detailed | [vit-b-16](https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/mPLUG/ViT-B-16.tar) | 6 | 6 | 12 | --M | [visualsitediary.detailed](https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/mPLUG/mplug_large_v2.pth) |

* Pre-train Image Datasets (Need to make a request for the images through the authors of each paper)
                                                                          
| | ACID | ACTV | SAFE | SODA | 
|------------------------|-------------------------------------------|------|------|------|
|image | 113K | 100K | 860K | 3M | 
|text | 567K | 769K | 860K | 3M |

* We share our captions on each image dataset in the `construction_dataset' folder.
  
## Requirements
* [PyTorch](https://pytorch.org/) version >= 1.11.0

* Install other libraries via
```
pip install -r requirements.txt
```

## Fine-tuning
                                                                                      
1. Download the Construction image dataset from the original paper.
  1.1. ACID (https://www.acidb.ca/dataset)
  1.2. ACTV (https://github.com/HannahHuanLIU/AEC-image-captioning)
  1.3. SAFE (https://doi.org/10.1061/JCEMD4.COENG-12096)
  1.4. SODA (https://doi.org/10.1016/j.autcon.2022.104499)  
2. Download and extract the provided dataset json files (our VSD Caption dataset).
3. Download language evalution tool([language_evalution](https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/mPLUG/language_evaluation.tar)).
4. In configs/caption_mplug_ALL.yaml, set the paths for the json files and the image paths.
5. Finetune the pre-trained visualsitediary_base or compact/detailed version model using 1 A100 GPU following our instruction.ipyb:
<pre>instruction.ipyb</pre> 
<pre>scripts/caption_vsd_base.sh</pre> 
<pre>sh scripts/caption_vsd_compact.sh</pre>  
<pre>sh scripts/caption_vsd_detail.sh</pre>  

                                                                   
## Demo instruction for Inference (using your own dataset)

## Citation
If you use our work, please cite:
```
@article{jung2023,
  title={VisualSiteDiary: A Detector-Free Vision Transformer Model for Captioning Photologs for Daily Construction Reporting},
  author={Jung, Yoonhwa and Cho, Ikhyun and Hsu, Shun-Shuing and Golparvar-Fard, Mani},
  journal={Automation in Construction},
  year={2023},
  note ={submitted}
}
```
## Acknowledgement

The implementation of VisualSiteDiary relies on resources from [mPLUG](https://github.com/alibaba/AliceMind/tree/main/mPLUG), and [S2-Transformer](https://github.com/zchoi/S2-Transformer). We thank the original authors for their open-sourcing.
