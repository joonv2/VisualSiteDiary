# VisualSiteDiary
An image captioning model built upon a pretrained ViT model (mPLUG) that provides human-readable captions to decipher daily progress and work activities from construction photologs
## Introduction
We present VisualSiteDiary that provides human-readable captions to decipher daily progress and work activities from construction photologs. To achieve high-quality descriptions from these photologs, our method: 1) incorporates pseudo region features, 2) utilizes high-level knowledge for pretraining the model, and 3) fine-tunes the model to consider different styles of captions accommodating different construction use-cases such as daily construction reporting. To validate the model and enable future research, we present VSD, a new comprehensive image captioning dataset that demonstrates various construction scenarios such as site work and super/sub-structure activities. Experiments show that \MethodName provides superior-quality captions compared to the state-of-the-art image captioning models on the VSD dataset. The dataset itself also offers more realistic yet challenging cases that need to be considered for real application on a project. We discuss how our method can be used for 1) daily construct reporting, and 2) image retrieval from existing photologs. Examples are also shared on how \MethodName enables variable style-caption generation from challenging construction images. 

<img src="VSD_framework.png" width="600"> 


## Pre-trained models and datasets

* Pre-trained models

 
 
|Model | Visual Backbone | Text Enc Layers | Fusion Layers | Text Dec Layers | #params | Download |
|------------------------|-------------------------------------------|------|------|------|------|-----|
|visualsitediary.total | [vit-b-16](https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/mPLUG/ViT-B-16.tar) | 6 | 6 | 12 | --M | [visualsitediary.total](https://drive.google.com/file/d/1CmEpPnHGS-pZw7XFi3WUokcmeuvhA5ib/view?usp=sharing) |
|visualsitediary.compact | [vit-b-16](https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/mPLUG/ViT-B-16.tar) | 6 | 6 | 12 | --M | [visualsitediary.compact](https://drive.google.com/file/d/1-fWTDclYFy4PaKVqIdfHZqphCeSfpStX/view?usp=sharing) |
|visualsitediary.detailed | [vit-b-16](https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/mPLUG/ViT-B-16.tar) | 6 | 6 | 12 | --M | [visualsitediary.detailed](https://drive.google.com/file/d/1lAjpJ_bJdUWfQDXGbNmn72_g_BfSLZOb/view?usp=sharing) |

* VSD Datasets (Need to make a request for the images through the authors of each paper)
                                                                          
| | ACID | ACTV | SAFE | SODA | 
|------------------------|-------------------------------------------|------|------|------|
|image | 4,000 | 964 | 1,762 | 1,089 | 
|text | 8,000 | 1,928 | 3,524 | 2,178 |

* We share our captions on each image dataset in the `construction_dataset' folder.
  
## Requirements
* [PyTorch](https://pytorch.org/) version >= 1.11.0

* Install other libraries via
```
pip install -r requirements.txt
```

## Fine-tuning
If you want to train your own model you can follow the instructions below.
                                                                                      
  1. Download the Construction image dataset from the original paper.
      - ACID (https://www.acidb.ca/dataset)
      - ACTV (https://github.com/HannahHuanLIU/AEC-image-captioning)
      - SAFE (https://doi.org/10.1061/JCEMD4.COENG-12096)
      - SODA (https://doi.org/10.1016/j.autcon.2022.104499)  
  2. Modify configs/VSD_all.yaml so the directories of the images and json files are correct.
  3. Download and upload ViT-B-16.tar to the main root (./)
  4. Download and upload mPLUG_base.pth to the main root (./)
<!-- 5. Upload --ours--.pth to the main root (./)
  - visualsitediary_total: Trained_with_both_captions.pth
  - visualsitediary_compact: Trained_with_lv_0.pth
  - visualsitediary_detail: Trained_with_lv_1.pth -->
  5. Download language evalution tool([language_evalution](https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/mPLUG/language_evaluation.tar)).
  6. Follow the steps 1~4 below.
<!-- 8. Finetune the pre-trained visualsitediary_base or compact/detailed version model using 1 A100 GPU following our instruction.ipyb:
<pre>instruction.ipyb</pre> 
<pre>scripts/caption_vsd_base.sh</pre> 
<pre>sh scripts/caption_vsd_compact.sh</pre>  
<pre>sh scripts/caption_vsd_detail.sh</pre>  -->


* Step 1. First, We begin with the HP process. Run the following script:
    ```
    python -c "import language_evaluation; language_evaluation.download('coco')"
    python run_VSD.py \
    --config ./configs/VSD_all.yaml \
    --checkpoint ./mPLUG_base.pth \
    --do_two_optim \
    --lr 1e-5 \
    --min_length 8 \
    --max_length 25 \
    --max_input_length 25 \
    --eval_start_epoch 0 \
    --save_for_HP True \
    --use_PR False
    ```
    * Make sure that the (datasets, json files, and model checkpoints) are in the correct directory.
    * The outputs will be saved at f'./output/result/train_loader_epoch_{i}.json'

* Step 2. Create the HP labels based on the model output from step 1 by running the following script.
    ```
    python -c "import language_evaluation; language_evaluation.download('coco')"
    python create_HP_labels.py \
    --source_prediction_train f'./output/result/train_loader_epoch_{i}.json' \
    --source_prediction_val f'./output/result/val_loader_epoch_{i}.json'  \
    --save_dir './construction_dataset/'
    ```
    * The HP labels will be saved at './construction_dataset/'.

* Step 3. Run HP by running the following script.
    ```
    python -c "import language_evaluation; language_evaluation.download('coco')"
    python run_VSD_HP.py \
    --config ./configs/VSD_HP.yaml \
    --checkpoint ./mPLUG_base.pth \
    --do_two_optim \
    --lr 1e-5 \
    --min_length 8 \
    --max_length 25 \
    --max_input_length 25 \
    --eval_start_epoch 0 \
    ```
    * The HP-pretrained model will be saved at './HP_pretrained.pth'

* Step 4. Finetune the HP-pretrained model with PR on the VSD image captioning dataset by running the following script:
    ```
    python -c "import language_evaluation; language_evaluation.download('coco')"
    python run_VSD.py \
    --config ./configs/VSD_all.yaml \
    --checkpoint ./HP_pretrained.pth \
    --do_two_optim \
    --lr 1e-5 \
    --min_length 8 \
    --max_length 25 \
    --max_input_length 25 \
    --eval_start_epoch 0 \
    --save_for_HP False \
    --use_PR True
    ```
                                                                   
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
