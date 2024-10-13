
import os
import json
import torch
import pickle
import itertools
import torch.distributed as dist
from definitions import LowShotConfig
from typing import List,Dict
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from transformers.models.vilt.processing_vilt import ViltProcessor
from transformers import BertTokenizerFast
from transformers import ViltConfig


class PIQA_Dataset_VILT(Dataset):
    def __init__(self,
                 VILT_ckpt_dir,
                 VILT_tokenizer,
                 data_dir:str,
                 split:str,
                 low_shot_config=LowShotConfig,):
        super().__init__()
        self.processor = ViltProcessor.from_pretrained(os.path.join(VILT_ckpt_dir,"vilt"))
        self.processor.tokenizer = BertTokenizerFast.from_pretrained(VILT_tokenizer)
        vilt_config = ViltConfig.from_pretrained(os.path.join(VILT_ckpt_dir,"vilt"))
        self.max_text_length = 40
        
        if split == "train":
            
            if low_shot_config.key:
                data_path = os.path.join(data_dir,"PIQA", "train.jsonl")
                label_path = os.path.join(data_dir,"PIQA","train-labels.lst")
                cached_data = os.path.join(data_dir,"PIQA","cached_data_file","piqa_train_2048shot.pkl")
            else:
                data_path = os.path.join(data_dir,"PIQA", "train.jsonl")
                label_path = os.path.join(data_dir,"PIQA","train-labels.lst")
                cached_data = os.path.join(data_dir,"PIQA","cached_data_file","piqa_train.pkl")
        elif split == "val":
            if low_shot_config.key:
                data_path = os.path.join(data_dir,"PIQA", "valid.jsonl")
                label_path = os.path.join(data_dir,"PIQA","valid-labels.lst")
                cached_data = os.path.join(data_dir,"PIQA","cached_data_file","piqa_val.pkl")
            else:
                data_path = os.path.join(data_dir,"PIQA","valid.jsonl")
                label_path = os.path.join(data_dir,"PIQA","valid-labels.lst")
                cached_data = os.path.join(data_dir,"PIQA","cached_data_file","piqa_val.pkl")
            
        self.mean_image = Image.open("Utils/coco_mean_image.png")
        self.mean_image =  self.mean_image.convert('RGB')
        
        if min(list(self.mean_image.size)) > 384:
            pil_transform = transforms.Resize(size=384, max_size=640)
            self.mean_image = pil_transform(self.mean_image)
        
        if os.path.exists(cached_data):
            self.data = pickle.load(open(cached_data,"rb"))
        else:
            
            data = []
            with open(data_path,"r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    line_data = json.loads(line)
                    data.append(line_data)
            
            labels = open(label_path, encoding="utf-8").read().splitlines()
            
            label_list = ["0", "1"]
            label = {label: i for i, label in enumerate(label_list)}
            
            self.data = []
            for idx,(dt,lb) in enumerate(zip(data,labels)):
                merged_text = [dt["goal"] + t_b for t_b in [dt["sol1"], dt["sol2"]]]
                
                labeled_data = {
                "example_id": idx,
                "text_a": dt["goal"],
                "text_b": [dt["sol1"], dt["sol2"]],
                "merged_text": merged_text,
                "label": label[lb],
                "description":"Multiple-Choice; text_a: Ctx; text_b: Ans"
                }
                self.data.append(labeled_data)
            pickle.dump(self.data,open(cached_data,"wb")) 
        self.n_samples = len(self.data)
        
    def __getitem__(self,index):
        batch = self.data[index]
        
        example_id = batch["example_id"]
        text_a = batch["text_a"]
        text_b = batch["text_b"]
        merged_text = batch["merged_text"]
        label = batch["label"]
        
        return {"example_id":example_id,
                "text_a":text_a,
                "text_b":text_b,
                "merged_text":merged_text,
                "label":label}
        
    def __len__(self):
        return self.n_samples
        
        
    def piqa_batch_collate(self,
                           batch:List[Dict]):
        """Collates each model input for all batch items into a single model input.

        Args:
            batch (List[Dict]): _description_
        """
        
        example_id = [x["example_id"] for x in batch]
        text_a = [x["text_a"] for x in batch]
        text_b = [x["text_b"] for x in batch]
        bs = len(batch)
        text_pairs = []
        for i in range(bs):
            for j in range(len(text_b[0])):
                each_pairs = []
                each_pairs.append(text_a[i])
                each_pairs.append(text_b[i][j])
                text_pairs.append(each_pairs)
        image = [self.mean_image for _ in batch]
        
        encodings = self.processor(images=image,
                                    text=text_pairs,
                                   padding=True,
                                   max_length=self.max_text_length,
                                   truncation=True,
                                   return_tensors='pt')
        
        
        label = [x["label"] for x in batch]
        label = torch.tensor(label)
        
        return {
            "encodings":encodings,
            "labels":label
        }