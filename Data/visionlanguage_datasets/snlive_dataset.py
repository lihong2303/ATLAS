
from io import BytesIO
import re
import torch
import os
import base64
import pickle
import jsonlines
import torch.distributed as dist
from torchvision import transforms
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from typing import List,Dict
from transformers.models.vilt.processing_vilt import ViltProcessor
from transformers import BertTokenizerFast
from transformers import ViltConfig


class SnliVeDataset_ViLT(Dataset):
    def __init__(
        self,
        split,
        low_shot_config,
        data_dir,
        VILT_ckpt_dir,
        VILT_tokenizer
    ):
        super().__init__()
        self.split = split
        self.data_dir = data_dir
        self.VILT_ckpt_dir = VILT_ckpt_dir
        self.VILT_tokenizer = VILT_tokenizer
        self.processor = ViltProcessor.from_pretrained(os.path.join(VILT_ckpt_dir,"vilt"))
        self.processor.tokenizer = BertTokenizerFast.from_pretrained(VILT_tokenizer)
        vilt_config = ViltConfig.from_pretrained(os.path.join(VILT_ckpt_dir,"vilt"))
        self.max_text_length = 30

        self.image_list = os.listdir(os.path.join(self.data_dir,"snli_ve_data","flickr30k_images"))
        
        if low_shot_config.key:
            if split == "train":
                self.cached_data_file = os.path.join(self.data_dir,"snli_ve_data","cached_data_file","snli_ve_train_2048shot.pkl")
            else:
                self.cached_data_file = os.path.join(self.data_dir,"snli_ve_data","cached_data_file","snli_ve_{}.pkl".format(split))
        else:
            data_path = os.path.join(self.data_dir,"snli_ve_data","snli_ve_{}.jsonl".format(split))
            self.cached_data_file = os.path.join(self.data_dir,"snli_ve_data","cached_data_file","snli_ve_{}.pkl".format(split))

        if os.path.exists(self.cached_data_file):
            self.dataset = pickle.load(open(self.cached_data_file,'rb'))
        else:
            categories = ['entailment', 'contradiction', 'neutral']
            cat2label = {cat: i for i, cat in enumerate(categories)}
            self.dataset = []
            json_lines = jsonlines.open(data_path)
            for line in json_lines:
                image_id = int(line['Flickr30K_ID'])
                hypothesis = str(line['sentence2'])
                gold_label = cat2label[line["gold_label"]]

                doc = {"image_id":image_id,
                    'hypothesis':hypothesis,
                    'label':gold_label}
                self.dataset.append(doc)
            pickle.dump(self.dataset,open(self.cached_data_file,"wb"))

        self.image_transform = transforms.Compose([
            transforms.Resize((384,384)),
            lambda image: image.convert('RGB')
        ])


    def __getitem__(self, index):

        example = self.dataset[index]

        hypothesis = example['hypothesis']
        image_id = example["image_id"]
        label = example["label"]

        image_path = os.path.join(self.data_dir,"snli_ve_data","flickr30k_images",str(image_id)+".jpg")
        if not os.path.exists(image_path):
            raise ValueError
        image = Image.open(image_path)
        image = self.image_transform(image)
        encodings = self.processor(images=image,
                                   text=hypothesis,
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
        
        example = {
            "encodings":encodings,
            "labels":label
        }
        return example
    
    def __len__(self):
        return len(self.dataset)

    def snlive_batch_collate_ViLT(self,batch:List[Dict]):
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
        label = [x["labels"] for x in batch]
        label = torch.tensor(label,dtype=torch.long)
        
        return {"encodings":encodings,
                "labels":label}
