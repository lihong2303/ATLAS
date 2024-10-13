
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

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)

class iNaturalist2019_Dataset_VILT(Dataset):
    def __init__(self,
                 data_dir:str,
                 split:str,
                 low_shot_config:LowShotConfig,
                 VILT_ckpt_dir,
                 VILT_tokenizer,):
        """
        Initiate the Dataset - loads all the image filenames and the corresponding labels.
        """
        super().__init__()
        self.low_shot_config = low_shot_config
        self.data_dir = data_dir
        self.split = split
        
        self.VILT_ckpt_dir = VILT_ckpt_dir
        self.VILT_tokenizer = VILT_tokenizer
        
        self.processor = ViltProcessor.from_pretrained(os.path.join(VILT_ckpt_dir,"vilt"))
        self.processor.tokenizer = BertTokenizerFast.from_pretrained(VILT_tokenizer)
        vilt_config = ViltConfig.from_pretrained(os.path.join(VILT_ckpt_dir,"vilt"))
        self.max_text_length = 7
        if low_shot_config.key and split == "train":
            self.cached_data_file = os.path.join(self.data_dir,"iNaturalist","cached_data_file","iNaturalist2019_{}_2048shot.pkl".format(split))
        else:
            self.cached_data_file = os.path.join(self.data_dir,"iNaturalist","cached_data_file","iNaturalist2019_{}.pkl".format(split))

        if os.path.exists(self.cached_data_file):
            self.dataset = pickle.load(open(self.cached_data_file,'rb'))
        else:
            print("Need preprocess data")
            if split in ["train","val"]:
                self.train_path = os.path.join(self.data_dir,"iNaturalist","train2019.json")
                with open(self.train_path) as f:
                    ann_data = json.load(f)

                all_img_fns = [a['file_name'] for a in ann_data['images']]
                all_labels = [a['category_id'] for a in ann_data['annotations']]

                dataset = [[] for _ in range(1010)]
                for label,fn in zip(all_labels,all_img_fns):
                    dataset.append([fn,label])
                
                train_dataset,val_dataset = [],[]
                for cls_data in dataset:
                    
                    n_train = len(cls_data) - int(len(cls_data) * 0.1)

                    random.seed(2022)
                    random.shuffle(cls_data)

                    train_cls_ds = cls_data[:n_train]
                    train_dataset.extend(train_cls_ds)
                    val_dataset.extend(cls_data[n_train:])

                pickle.dump(train_dataset,open(os.path.join(self.data_dir,"cached_data_file","iNaturalist2019_train.pkl"),"wb"))
                pickle.dump(val_dataset,open(os.path.join(self.data_dir,"cached_data_file","iNaturalist2019_val.pkl"),"wb"))
                if split == "train":
                    self.dataset = train_dataset
                else:
                    self.dataset = val_dataset
            elif split == "test":
                self.test_path = os.path.join(self.data_dir,"val2019.json")
                with open(self.test_path) as f:
                    ann_data = json.load(f)

                all_img_fns = [a['file_name'] for a in ann_data['images']]
                all_labels = [a['category_id'] for a in ann_data['annotations']]

                test_dataset = []
                for label,fn in zip(all_labels,all_img_fns):
                    test_dataset.append([fn,label])
                pickle.dump(test_dataset,open(os.path.join(self.data_dir,"cached_data_file","iNaturalist2019_test.pkl"),"wb"))
                self.dataset = test_dataset
            
        self.image_transform = transforms.Compose([
            transforms.CenterCrop(size=384),
            lambda image: image.convert('RGB')
        ])
        
        self.n_samples = len(self.dataset)
        
    def __getitem__(self,index):
        filename,label = self.dataset[index]
        image = Image.open(os.path.join(self.data_dir,"iNaturalist",filename))
        image = self.image_transform(image)
        
        text = "This is an image."
        
        encodings = self.processor(images=image,
                                   text=text,
                                   padding=True,
                                   max_length=self.max_text_length,
                                   truncation=True,
                                   return_tensors='pt')
        expand_tensor = torch.zeros((encodings["input_ids"].size(0),self.max_text_length - encodings["input_ids"].size(1)),dtype=torch.int64)
        encodings["input_ids"] = torch.cat([encodings["input_ids"],expand_tensor],dim=1)
        expand_tensor = torch.zeros((encodings["token_type_ids"].size(0),self.max_text_length - encodings["token_type_ids"].size(1)),dtype=torch.int64)
        encodings["token_type_ids"] = torch.cat([encodings["token_type_ids"],expand_tensor],dim=1)
        expand_tensor = torch.zeros((encodings["attention_mask"].size(0),self.max_text_length - encodings["attention_mask"].size(1)),dtype=torch.int64)
        encodings["attention_mask"] = torch.cat([encodings["attention_mask"],expand_tensor],dim=1)
        
        return {"encodings":encodings,
                "label":label}
    
    def __len__(self):
        return self.n_samples
        
        
    def load_image(self,image_path):
        image = Image.open(image_path)
        image = self.image_transform(image)
        return image
        
    def iNatural2019_batch_collate(self,batch:List[Dict]):
        """Collates each model input for all batch items into a single model input.

        Args:
            batch (List[Dict]): _description_
        """
        
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
