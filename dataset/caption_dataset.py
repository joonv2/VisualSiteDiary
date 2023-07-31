import json
import numpy as np
import time
import logging
import os
import random

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile

import oss2
from io import BytesIO
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption

def decode_int32(ann):
    ann = str(ann)
    server = str(int(ann[-1]) + 1)
    id_ = "0"*(9-len(ann[:-1]))+ann[:-1]
    assert len(id_) == 9
    ann = server+"/"+id_
    return ann

class re_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}   
        
        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = pre_caption(ann['caption'], self.max_words) 

        return image, caption, self.img_ids[ann['image_id']]
    
    

class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words 
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption,self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                                    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.ann[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index
      
class nocaps_dataset(Dataset):
    def __init__(self, ann_file, transform, root_path, max_words=30, read_local_data=True, is_train=True, add_object=False):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.max_words = max_words
        self.read_local_data = read_local_data
        self.root_path = root_path
        self.add_object = add_object
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        """ 
        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)
        """
        image_id = ann['img_id'] 
        if self.read_local_data:
            image_path = os.path.join(self.root_path, ann['image'])
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        else:
            while not self.bucket.object_exists("mm_feature/"+ann['image']):
                index = 0 if index == (len(self) - 1) else index + 1
                ann = self.ann[index]
            while True:
                try:
                    # file_str = bucket.get_object(file_path)
                    # file_buf = io.BytesIO()
                    # file_buf.write(file_str.read())
                    # file_buf.seek(0)
                    # file_buf = BytesIO(bucket.get_object(file_path).read())
                    # img_info = np.load(file_buf)
                    # file_buf.close()
                    image = self.bucket.get_object("mm_feature/"+ann['image'])
                    image = BytesIO(image.read())
                    image = Image.open(image).convert('RGB')
                    image = self.transform(image)
                except:
                    #logging.info("Get image:{} from oss failed, retry.".format(ann['image']))
                    index = 0 if index == (len(self) - 1) else index + 1
                    ann = self.ann[index]
                    continue
                break
                
        return image, image_id

class coco_dataset(Dataset):
    def __init__(self, ann_file, transform, root_path, max_words=30, read_local_data=True, is_train=True, add_object=False):
        self.ann = []
        for f in ann_file:
            self.ann.append(json.load(open(f,'r')))
        self.transform = transform
        self.max_words = max_words
        self.read_local_data = read_local_data
        self.root_path = root_path
        self.ann_new = []
        self.add_object = add_object
        
        if 'train' in ann_file[0]:
            for i, dataset in enumerate(self.ann):
                img_id_to_image_path = {}     # input = image_id, output = image path.
                img_id_to_gold_captions = {}  # input = image_id, output = collection of all gold captions.

                for item in dataset['images']:
                    img_id_to_image_path[item['id']] = os.path.join(self.root_path[i], item['file_name'])

                for item in dataset['annotations']:
                    if item['image_id'] in img_id_to_gold_captions.keys():
                        img_id_to_gold_captions[item['image_id']].append(item['caption'].lower())
                    else:
                        img_id_to_gold_captions[item['image_id']] = [item['caption'].lower()]

                object_label = ""    # Not used.

                for item in dataset['annotations']:
                    caption_ = item['caption'].lower()
                    if caption_[-1] != '.':
                        caption_ += '.'
                    elif caption_[-1] == ' ':
                        caption_[-1] = '.'
                    self.ann_new.append({"image": img_id_to_image_path[item['image_id']],
                                         "caption": caption_, 
                                         "gold_caption": img_id_to_gold_captions[item['image_id']],
                                         "object_label": object_label})
                    
        elif 'val' in ann_file[0]:
            for i, dataset in enumerate(self.ann):
                used_img_id = []
                img_id_to_image_path = {}     # input = image_id, output = image path.
                img_id_to_gold_captions = {}  # input = image_id, output = collection of all gold captions.

                for item in dataset['images']:
                    img_id_to_image_path[item['id']] = os.path.join(self.root_path[i], item['file_name'])

                for item in dataset['annotations']:
                    if item['image_id'] in img_id_to_gold_captions.keys():
                        img_id_to_gold_captions[item['image_id']].append(item['caption'].lower())
                    else:
                        img_id_to_gold_captions[item['image_id']] = [item['caption'].lower()]

                object_label = ""    # Not used.

                for item in dataset['annotations']:
                    if item['image_id'] in used_img_id:
                        continue
                    else:
                        caption_ = item['caption'].lower()
                        if caption_[-1] != '.':
                            caption_ += '.'
                        elif caption_[-1] == ' ':
                            caption_[-1] = '.'
                        
                        used_img_id.append(item['image_id'])
                        self.ann_new.append({"image": img_id_to_image_path[item['image_id']],
                                             "caption": caption_, 
                                             "gold_caption": img_id_to_gold_captions[item['image_id']],
                                             "object_label": object_label})
            
            
        self.ann = self.ann_new
        del self.ann_new
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        caption = ann['caption']
        image_id = ann['image']
        object_label = ann['object_label']
        if self.read_local_data:
#             image_path = os.path.join(self.root_path, ann['image'])
            image_path = ann['image']
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        else:
            while not self.bucket.object_exists("mm_feature/"+ann['image']):
                index = 0 if index == (len(self) - 1) else index + 1
                ann = self.ann[index]
            while True:
                try:
                    image = self.bucket.get_object("mm_feature/"+ann['image'])
                    image = BytesIO(image.read())
                    image = Image.open(image).convert('RGB')
                    image = self.transform(image)
                except:
                    #logging.info("Get image:{} from oss failed, retry.".format(ann['image']))
                    index = 0 if index == (len(self) - 1) else index + 1
                    ann = self.ann[index]
                    continue
                break
                
        return image, caption, object_label, image_id, ann["gold_caption"]
    
    
class coco_dataset_HP(Dataset):
    def __init__(self, ann_file, transform, max_words=30, read_local_data=True, is_train=True, add_object=False):
        self.ann = []
        for f in ann_file:
            self.ann.append(json.load(open(f,'r')))
        self.transform = transform
        self.max_words = max_words
        self.read_local_data = read_local_data
        self.ann_new = []
        self.add_object = add_object
        
        if 'train' in ann_file[0]:
            for i, dataset in enumerate(self.ann):
                for item in dataset:
                    self.ann_new.append({"image": item['img_path'],
                                         "HP_label": item['HP_label']})
                    
        elif 'val' in ann_file[0]:
            for i, dataset in enumerate(self.ann):
                for item in dataset:
                    self.ann_new.append({"image": item['img_path'],
                                         "HP_label": item['HP_label']})
            
        self.ann = self.ann_new
        del self.ann_new
        
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        image_id = ann['image']
        if self.read_local_data:
#             image_path = os.path.join(self.root_path, ann['image'])
            image_path = ann['image']
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        else:
            while not self.bucket.object_exists("mm_feature/"+ann['image']):
                index = 0 if index == (len(self) - 1) else index + 1
                ann = self.ann[index]
            while True:
                try:
                    image = self.bucket.get_object("mm_feature/"+ann['image'])
                    image = BytesIO(image.read())
                    image = Image.open(image).convert('RGB')
                    image = self.transform(image)
                except:
                    #logging.info("Get image:{} from oss failed, retry.".format(ann['image']))
                    index = 0 if index == (len(self) - 1) else index + 1
                    ann = self.ann[index]
                    continue
                break
                
        return image, ann["HP_label"]
        
    
class pretrain_dataset_4m(Dataset):
    def __init__(self, ann_file, transform, max_words=30, read_local_data=True, image_root="", epoch=None):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.max_words = max_words
        self.read_local_data = read_local_data
        self.image_root = image_root
        if not self.read_local_data:
            bucket_name = "xxxxx"
            auth = oss2.Auth("xxxxx", "xxxxxx")
            self.bucket = oss2.Bucket(auth, "xxxxx", bucket_name)
        
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)

        if self.read_local_data:
            image = Image.open(os.path.join(self.image_root, ann['image'])).convert('RGB')
            image = self.transform(image)
        else:
            while True:
                try:
                    image = self.bucket.get_object("mm_feature/"+ann['image'])
                    image = BytesIO(image.read())
                    image = Image.open(image).convert('RGB')
                    image = self.transform(image)
                except:
                    #logging.info("Get image:{} from oss failed, retry.".format(ann['image']))
                    time.sleep(0.1)
                    index = 0 if index == (len(self) - 1) else index + 1
                    ann = self.ann[index]
                    continue
                break
                
        return image, caption
