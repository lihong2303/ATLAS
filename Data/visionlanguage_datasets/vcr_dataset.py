from io import BytesIO
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


GENDER_NEUTRAL_NAMES = ['Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry', 'Jody', 'Kendall',
                        'Skyler', 'Frankie', 'Pat', 'Quinn', 'Morgan', 'Finley', 'Harley', 'Robbie', 'Sidney', 'Tommie',
                        'Ashley', 'Carter', 'Adrian', 'Clarke', 'Logan', 'Mickey', 'Nicky', 'Parker', 'Tyler',
                        'Reese', 'Charlie', 'Austin', 'Denver', 'Emerson', 'Tatum', 'Dallas', 'Haven', 'Jordan',
                        'Robin', 'Rory', 'Bellamy', 'Salem', 'Sutton', 'Gray', 'Shae', 'Kyle', 'Alex', 'Ryan',
                        'Cameron', 'Dakota']

def process_list(mytext, objects):
    text = ''
    for element in mytext:
        if(type(element) == list): 
            for subelement in element:
                if(objects[int(subelement)] == 'person'):
                    temporal_text = GENDER_NEUTRAL_NAMES[int(subelement)]
                else:
                    temporal_text = 'the gray ' + str(objects[int(subelement)]).strip()
        elif(type(element) == int):
            if(objects[int(element)] == 'person'):
                temporal_text = GENDER_NEUTRAL_NAMES[int(subelement)]
            else:
                temporal_text = 'the gray ' + str(objects[int(subelement)])
        else:
            temporal_text = element
        text += temporal_text + ' '
    return text


class VCRDataset_ViLT(Dataset):
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
        self.max_text_length = 40

        if low_shot_config.key:
            if split == "train":
                self.annotations_file = os.path.join(self.data_dir,"VCR","train.jsonl")
                self.cached_data_file = os.path.join(self.data_dir,"VCR","cached_data_file","VCR_train_fewshot.pkl")
            else:
                self.annotations_file = os.path.join(self.data_dir,"VCR","train.jsonl")
                self.cached_data_file = os.path.join(self.data_dir,"VCR","cached_data_file","VCR_{}.pkl".format(split))
        else:
            self.annotations_file = os.path.join(self.data_dir,"VCR","{}.jsonl".format(split))
            self.cached_data_file = os.path.join(self.data_dir,"VCR","cached_data_file","VCR_{}.pkl".format(split))

        if os.path.exists(self.cached_data_file):
            self.dataset = pickle.load(open(self.cached_data_file,'rb'))
        else:
            idx = 0
            self.dataset = []
            json_lines = jsonlines.open(self.annotations_file)
            for line in json_lines:
                image_fn_path = line["img_fn"]
                image_path = os.path.join(self.data_dir,"VCR","vcr1images",image_fn_path)
                multichoice_texts = []
                objects = line["objects"]

                question = process_list(line["question"],objects) # question
                for answer in line['answer_choices']:                        
                    answer1 = process_list(answer, objects)
                    text = question + ' [SEP] ' + answer1
                    multichoice_texts.append(text)
                label = int(line['answer_label']) ##number
                doc = {"image_path":image_path,
                    "texts":multichoice_texts,
                    "label":label}
                if low_shot_config.key and split == "train":
                    if idx % 20 == 0:
                        self.dataset.append(doc)
                    idx += 1
                else:
                    self.dataset.append(doc)
            pickle.dump(self.dataset,open(self.cached_data_file,'wb'))
        self.n_samples = len(self.dataset)

        self.image_transform = transforms.Compose([
            transforms.Resize((384,384)),
            lambda image: image.convert('RGB'),
            # transforms.ToTensor(),
        ])

    def __getitem__(self,index):
        example = self.dataset[index]

        image_path = example["image_path"]
        texts = example["texts"]
        label = example["label"]

        image = Image.open(image_path)
        image = self.image_transform(image)
        encodings = self.processor(images=image,
                                   text=texts,
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
    
    def VCR_batch_collate(self,batch):

        encodings = {}
        input_ids = [x["encodings"]["input_ids"].unsqueeze(0) for x in batch]
        input_ids = torch.cat(input_ids,dim=0)
        encodings["input_ids"] = input_ids
        token_type_ids = [x["encodings"]["token_type_ids"].unsqueeze(0) for x in batch]
        token_type_ids = torch.cat(token_type_ids,dim=0)
        encodings["token_type_ids"] = token_type_ids
        
        attention_mask = [x["encodings"]["attention_mask"].unsqueeze(0) for x in batch]
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