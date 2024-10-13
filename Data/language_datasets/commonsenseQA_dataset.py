
import re
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


class CommonsenseQA_VILT(Dataset):
    def __init__(self,
                 VILT_ckpt_dir,
                 VILT_tokenizer,
                 data_dir:str,
                 split:str,
                 low_shot_config=LowShotConfig,) -> None:
        super().__init__()
        self.processor = ViltProcessor.from_pretrained(os.path.join(VILT_ckpt_dir,"vilt"))
        self.processor.tokenizer = BertTokenizerFast.from_pretrained(VILT_tokenizer)

        self.max_text_length = 40
        if split == "train":
            if low_shot_config.key:
                # Update
                data_path = os.path.join(data_dir,"CommonsenseQA","train_rand_split.jsonl")
                cached_path = os.path.join(data_dir,"CommonsenseQA","cached_data_file","train_{}shot.jsonl".format(low_shot_config.num_low_shot))
            else:
                data_path = os.path.join(data_dir,"CommonsenseQA","train_rand_split.jsonl")
        elif split == "val":
            data_path = os.path.join(data_dir,"CommonsenseQA","dev_rand_split.jsonl")

        self.mean_image = Image.open("Utils/coco_mean_image.png")
        self.mean_image =  self.mean_image.convert('RGB')

        data = []
        with open(data_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                line = json.loads(line)
                data.append(line)

        self.data = []
        label_list = ["A", "B", "C", "D", "E"]
        label_map = {label: i for i, label in enumerate(label_list)}
        if low_shot_config.key and split =="train":
            if os.path.exists(cached_path):
                self.data = pickle.load(open(cached_path,'rb'))
            else:
                num_class = {i:0 for i in range(5)}
                for idx,dt in enumerate(data):
                    text_a = dt['question']['stem'],
                    text_b = [ch['text'] for ch in dt['question']['choices']]
                    merged_text = [f'{text_a} [SEP] {t_b}' for t_b in text_b]
                    label = label_map[dt["answerKey"]]
                    labeled_data = {
                        "example_id":idx,
                        "text_a":text_a,
                        "text_b":text_b,
                        "merged_text":merged_text,
                        "label":label,
                    }
                    if num_class[label] < low_shot_config.num_low_shot:
                        self.data.append(labeled_data)
                        num_class[label] += 1
                pickle.dump(self.data,open(cached_path,'wb'))
        else:
            for idx,dt in enumerate(data):
                text_a = dt['question']['stem'],
                text_b = [ch['text'] for ch in dt['question']['choices']]
                merged_text = [f'{text_a} [SEP] {t_b}' for t_b in text_b]
                label = label_map[dt["answerKey"]]
                labeled_data = {
                    "example_id":idx,
                    "text_a":text_a,
                    "text_b":text_b,
                    "merged_text":merged_text,
                    "label":label,
                }
                self.data.append(labeled_data)
        
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

    def CommonsenseQA_batch_collate(self,batch):
        text_a = [x["text_a"][0] for x in batch]
        text_b = [x["text_b"] for x in batch]
        bs = len(batch)
        # text_b = list(itertools.chain(*text_b)) #texts_b (n_choice, bs) -> (n_choice*bs,)
        text_pairs = []
        for i in range(bs):
            for j in range(len(text_b[0])):
                each_pairs = []
                word_a = re.findall('\w+',text_a[i])
                word_b = re.findall('\w+',text_b[i][j])
                if len(word_a) + len(word_b) > 40:
                    word_a = word_a[(len(word_b) + len(word_a) - 40):]
                each_pairs.append(' '.join(word_a))
                each_pairs.append(' '.join(word_b))
                text_pairs.append(each_pairs)
        
        image = [self.mean_image for _ in batch]
        encodings = self.processor(images=image,
                                   text=text_pairs,
                                   padding=True,
                                   max_length=40,
                                   truncation=True,
                                   return_tensors='pt')
        label = [x["label"] for x in batch]
        label = torch.tensor(label)

        return{"encodings":encodings,
               "labels":label}
