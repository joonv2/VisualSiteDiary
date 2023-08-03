import io
import json
import os

from torchvision import models
from torchvision import transforms
from PIL import Image
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

# VSD model
import ruamel.yaml as yaml
import torch
import torch.nn as nn
from models.model_caption_mplug import VSD
from models.vit import resize_pos_embed
from models.tokenization_bert import BertTokenizer

from summarizer import Summarizer
# from transformers import pipeline


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/")
@cross_origin()
def helloWorld():
  return "Hello, cross-origin-world!"


use_PR = True
device = torch.device('cuda')
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
config = yaml.load(open('./configs/VSD_all.yaml', 'r'), Loader=yaml.Loader)
config["min_length"] = 8
config["max_length"] = 25
config["add_object"] = False
config["beam_size"] = 5
config['text_encoder'] = "bert-base-uncased"
config['text_decoder'] = "bert-base-uncased"
model = VSD(config=config, tokenizer=tokenizer, use_PR=use_PR)
model = model.to(device)
'''
====== Modify checkpoint path ====== 
'''
checkpoint_name = f'visualsitediary.total'
checkpoint = torch.load(f"{checkpoint_name}.pth", map_location='cpu')
print(f"load checkpoint from {checkpoint_name}.pth")
state_dict = checkpoint['model']

# reshape positional embedding to accomodate for image resolution change
if config["clip_name"] == "ViT-B-16":
    num_patches = int(config["image_res"] * config["image_res"]/(16*16))
elif config["clip_name"] == "ViT-L-14":
    num_patches = int(config["image_res"] * config["image_res"]/(14*14))
pos_embed = nn.Parameter(torch.zeros(num_patches + 1, 768).float())

pos_embed = resize_pos_embed(
    state_dict['visual_encoder.visual.positional_embedding'].unsqueeze(0),
    pos_embed.unsqueeze(0))
state_dict['visual_encoder.visual.positional_embedding'] = pos_embed
msg = model.load_state_dict(state_dict, strict=False)
print(msg)

model.eval()

normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
input_transform = transforms.Compose([
    transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    normalize,])

def transform_image(image_bytes):
    images = []
    for ele in image_bytes:
        image = Image.open(io.BytesIO(ele)).convert('RGB')
        images.append(input_transform(image).to(device))
    return torch.stack(images)

def get_prediction(images):
    pred_captions = []
    for image in images:
        topk_id, _ = model(image.unsqueeze(0), "", "", train=False)
        pred = tokenizer.decode(topk_id[0][0]).replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").strip()
        pred_captions.append(pred)

    return pred_captions

bert_model = Summarizer()
@app.route('/summary/', methods=['POST'])
def summary():
    if request.method == 'POST':
        output_dict = {'summarized_caption': '', 'msg': ''}
        pred_captions = request.form['pred_captions']
        print(pred_captions)
        pred_captions = pred_captions.replace('\n', ' ')
        result = bert_model(pred_captions, min_length=20)
        print(result)
        result = ''.join(result)
        output_dict['summarized_caption'] = result
        return jsonify(output_dict)

@app.route('/total/', methods=['POST'])
def total_predict():
    if request.method == 'POST':
        output_dict = {'pred_captions': [], 'msg': ''}
        # img_files = request.files.getlist('imgs')
        img_files = [request.files['img']]
        img_bytes = [file.read() for file in img_files]
        imgs = transform_image(img_bytes)
        pred_captions = get_prediction(imgs)[0]
        output_dict['pred_captions'] = pred_captions
        return jsonify(output_dict)
    
# @app.route('/compact', methods=['POST'])
# def compact_predict():

    
# @app.route('/detailed', methods=['POST'])
# def detailed_predict():


if __name__ == '__main__':
    app.run()
