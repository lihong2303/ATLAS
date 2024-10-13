
import os
import torch
import logging
import json
import pickle
import random
import numpy as np
from PIL import Image
from io import BytesIO
from typing import List,Dict
from definitions import LowShotConfig
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers.models.vilt.processing_vilt import ViltProcessor
from transformers import BertTokenizerFast
from transformers import ViltConfig


class Places365_Dataset_VILT(Dataset):
    def __init__(self,
                 data_dir:str,
                 split:str,
                 low_shot_config:LowShotConfig,
                 VILT_ckpt_dir,
                 VILT_tokenizer,):
        super().__init__()
        self.low_shot_config = low_shot_config
        self.data_dir = data_dir
        self.split = split
        self.VILT_ckpt_dir = VILT_ckpt_dir
        self.VILT_tokenizer = VILT_tokenizer
        
        self.processor = ViltProcessor.from_pretrained(os.path.join(VILT_ckpt_dir,"vilt"))
        self.processor.tokenizer = BertTokenizerFast.from_pretrained(VILT_tokenizer)
        self.max_text_length = 7
        label_list = os.listdir(os.path.join(self.data_dir, "Places365","places365_standard","val"))
        map_label = {name:index for index,name in enumerate(label_list)}

        if low_shot_config.key and split == "train":
            self.data_path = os.path.join(self.data_dir, "Places365","places365_standard","{}.txt".format(split))
            self.cached_data_path = os.path.join(self.data_dir,"Places365","cached_data_file","Places365_{}_{}shot.pkl".format(split,low_shot_config.num_low_shot))
        else:
            self.data_path = os.path.join(self.data_dir, "Places365","places365_standard","{}.txt".format(split))
        
        with open(self.data_path,'r') as f:
            data = f.readlines()
        random.shuffle(data)
        if low_shot_config.key and split == "train":
            if os.path.exists(self.cached_data_path):
                load_data = pickle.load(open(self.cached_data_path,'rb'))
                self.dataset = load_data[0]
                self.labels = load_data[1]
            else:
                num_class = {i:0 for i in range(365)}
                self.dataset = []
                self.labels = []
                for line in data:
                    label = map_label[line.split("/")[1]]
                    if num_class[label] < low_shot_config.num_low_shot:
                        self.dataset.append(line.strip())
                        self.labels.append(label)
                        num_class[label] += 1
                save_data = []
                save_data.append(self.dataset)
                save_data.append(self.labels)
                pickle.dump(save_data,open(self.cached_data_path,'wb'))
        else:
            self.dataset = [line.strip() for line in data]
            self.labels = [map_label[line.split("/")[1]] for line in data]
        self.image_transform = transforms.Compose([
            transforms.CenterCrop(size=384),
            lambda image: image.convert('RGB')
        ])
        
        self.n_samples = len(self.dataset)

    def __getitem__(self,index):

        image_path = self.dataset[index]
        label = self.labels[index]

        img_path = os.path.join(self.data_dir, "Places365","places365_standard",image_path)
        image = Image.open(img_path)
        image = self.image_transform(image)

        text = "This is an image."
        
        encodings = self.processor(images=image,
                                   text=text,
                                   padding=True,
                                   max_length=self.max_text_length,
                                   truncation=True,
                                   return_tensors='pt')
        return {"encodings":encodings,
                "label":label}

    def __len__(self):
        return self.n_samples
    
    def Places365_batch_collate(self,batch):
        encodings = {}
        input_ids = [x["encodings"]["input_ids"] for x in batch]
        input_ids = torch.cat(input_ids,dim=0)
        encodings["input_ids"] = input_ids
        token_type_ids = [x["encodings"]["token_type_ids"] for x in batch]
        token_type_ids = torch.cat(token_type_ids,dim=0)
        encodings["token_type_ids"] = token_type_ids
        
        attention_mask = [x["encodings"]["attention_mask"] for x in batch]
        attention_mask = torch.cat(attention_mask,dim=0)
        encodings["attention_mask"] = attention_mask
        
        pixel_values = [x["encodings"]["pixel_values"] for x in batch]
        pixel_values = torch.cat(pixel_values,dim=0)
        encodings["pixel_values"] = pixel_values
        
        pixel_mask = [x["encodings"]["pixel_mask"] for x in batch]
        pixel_mask = torch.cat(pixel_mask,dim=0)
        encodings["pixel_mask"] = pixel_mask
        
        labels = [x["label"] for x in batch]
        batch_labels = torch.tensor(labels,dtype=torch.long)
            
        return {"encodings":encodings,
                "labels":batch_labels}
