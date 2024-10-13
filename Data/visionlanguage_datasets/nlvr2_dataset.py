

import torch
import os
import pickle
import jsonlines
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from transformers.models.vilt.processing_vilt import ViltProcessor
from transformers import BertTokenizerFast
from transformers import ViltConfig


class NLVR2Dataset_ViLT(Dataset):
    def __init__(
        self,
        split,
        low_shot_config,
        data_dir,
        VILT_ckpt_dir,
        VILT_tokenizer,
        max_src_length=80,
        max_tgt_length=30,
        patch_image_size=224,
        add_caption=False,
        constraint_trie=None,
        prompt_type="none"
    ):
        super().__init__()
        self.split = split
        self.data_dir = data_dir
        self.VILT_ckpt_dir = VILT_ckpt_dir
        self.VILT_tokenizer = VILT_tokenizer
        self.processor = ViltProcessor.from_pretrained(os.path.join(VILT_ckpt_dir,"vilt"))
        self.processor.tokenizer = BertTokenizerFast.from_pretrained(VILT_tokenizer)
        vilt_config = ViltConfig.from_pretrained(os.path.join(VILT_ckpt_dir,"vilt"))
        self.max_text_length = 40

        if split == "train" and low_shot_config.key:
            self.annotation_file = os.path.join(self.data_dir,"NLVR2","data","train.json")
            self.cached_data_file = os.path.join(self.data_dir,"NLVR2", "cached_data_file","NLVR2_{}.pkl".format(low_shot_config.num_low_shot))
        else:
            self.annotation_file = os.path.join(self.data_dir,"NLVR2","data","{}.json".format(split))
        
        if low_shot_config.key and split == 'train':
            if os.path.exists(self.cached_data_file):
                self.dataset = pickle.load(open(self.cached_data_file,'rb'))
            else:
                self.label_list = {i:0 for i in range(2)}
                self.image_dir = os.path.join(self.data_dir,"NLVR2","images",split)
                self.dataset = []

                with jsonlines.open(self.annotation_file) as reader:
                    for annotation in reader:
                        example = {}
                        example["id"] = annotation["identifier"]
                        example["image_id_0"] = os.path.join(self.image_dir, (
                            "-".join(annotation["identifier"].split("-")[:-1]) + "-img0.png"
                        ))
                        example["image_id_1"] = os.path.join(self.image_dir, (
                            "-".join(annotation["identifier"].split("-")[:-1]) + "-img1.png"
                        ))
                        example["sentence"] = str(annotation["sentence"])
                        example["labels"] = 0 if str(annotation["label"]) == "False" else 1
                        if self.label_list[example["labels"]] < low_shot_config.num_low_shot:
                            self.label_list[example["labels"]] += 1
                            self.dataset.append(example)
                with open(self.cached_data_file, 'wb') as f:
                    pickle.dump(self.dataset,f)
        else:
            self.image_dir = os.path.join(self.data_dir,"NLVR2","images",split)
            self.dataset = []

            with jsonlines.open(self.annotation_file) as reader:
                for annotation in reader:
                    example = {}
                    example["id"] = annotation["identifier"]
                    example["image_id_0"] = os.path.join(self.image_dir, (
                        "-".join(annotation["identifier"].split("-")[:-1]) + "-img0.png"
                    ))
                    example["image_id_1"] = os.path.join(self.image_dir, (
                        "-".join(annotation["identifier"].split("-")[:-1]) + "-img1.png"
                    ))
                    example["sentence"] = str(annotation["sentence"])
                    example["labels"] = 0 if str(annotation["label"]) == "False" else 1
                    self.dataset.append(example)
            
        self.n_samples = len(self.dataset)
        self.image_transform = transforms.Compose([
            transforms.Resize((384,384)),
            lambda image: image.convert('RGB')
        ])
    def __getitem__(self,index):
        example = self.dataset[index]

        img1 = Image.open(example["image_id_0"])
        image1 = self.image_transform(img1)
        img2 = Image.open(example["image_id_1"])
        image2 = self.image_transform(img2)

        image = [image1,image2]

        text = example["sentence"]

        label = example["labels"]

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
                "labels":label}
    def __len__(self):
        return len(self.dataset)
    
    def NLVR2_batch_collate(self,batch):

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
        
        pixel_values = [x["encodings"]["pixel_values"].unsqueeze(0) for x in batch]
        pixel_values = torch.cat(pixel_values,dim=0)
        encodings["pixel_values"] = pixel_values
        
        pixel_mask = [x["encodings"]["pixel_mask"].unsqueeze(0) for x in batch]
        pixel_mask = torch.cat(pixel_mask,dim=0)
        encodings["pixel_mask"] = pixel_mask
        label = [x["labels"] for x in batch]
        label = torch.tensor(label,dtype=torch.long)
        return {"encodings":encodings,
                "labels":label}
