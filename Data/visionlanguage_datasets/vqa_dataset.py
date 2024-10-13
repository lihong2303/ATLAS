
import os
import json
import torch
import pickle
import logging
from PIL import Image
from definitions import LowShotConfig
from collections import defaultdict
from typing import Any,Dict,List
from torch.utils.data import Dataset
from torchvision import transforms
from transformers.models.vilt.processing_vilt import ViltProcessor
from transformers import BertTokenizerFast
from transformers import ViltConfig

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)


class VQADataset_ViLT(Dataset):
    def __init__(self,
                 data_dir:str,
                 split:str,
                 low_shot_config:LowShotConfig,
                 VILT_ckpt_dir,
                 VILT_tokenizer,
                 **kwargs:Any):
        """Initializes the VQADataset - load all the questions (and converts to input IDs using the tokenizer, if provided)
        and answers (including converting each to a numeric label, and a score based on occurence from annotators.)

        Args:
            data_dir (str): _description_
            images_dataset (MSCOCOImagesDataset): instances of MSCOCOImageDataset, that is used to retrieve the MS-COCO image for each question.
            split (str): _description_
        """
        
        self.data_dir = data_dir
        self.split = split
        self.VILT_ckpt_dir = VILT_ckpt_dir
        self.VILT_tokenizer = VILT_tokenizer
        
        self.processor = ViltProcessor.from_pretrained(os.path.join(VILT_ckpt_dir,"vilt"))
        self.processor.tokenizer = BertTokenizerFast.from_pretrained(VILT_tokenizer)
        vilt_config = ViltConfig.from_pretrained(os.path.join(VILT_ckpt_dir,"vilt"))
        self.max_text_length = 28
        
        
        self.image_transform = transforms.Compose([
            transforms.CenterCrop(size=384),
            lambda image: image.convert('RGB'),
            # transforms.ToTensor(),
        ])
        if low_shot_config.key:
            if split == "train":
                self.cached_data_file = os.path.join(data_dir,'VQA', 'cached_data_file', 'vqa_{}_2048shot.pkl'.format(split))
            else:
                self.cached_data_file = os.path.join(data_dir,'VQA', 'cached_data_file', 'vqa_{}.pkl'.format(split))
            self.image_file = os.path.join(data_dir,"VQA",'{}2014'.format(split))
            self.ans2label_file = os.path.join(data_dir,'VQA','answer_dict.json')
        else:
            self.cached_data_file = os.path.join(data_dir,'VQA', 'cached_data_file', 'vqa_{}.pkl'.format(split))
            self.image_file = os.path.join(data_dir,'VQA','{}2014'.format(split))
            self.ans2label_file = os.path.join(data_dir,'VQA','answer_dict.json')
        #Create image path list
        image_filenames = os.listdir(self.image_file)
        self.imageid2filename = {}
        for fn in image_filenames:
            fnt = fn.split('_')[-1]
            image_id = int(fnt.strip('.jpg'))
            self.imageid2filename[image_id] = os.path.join(self.image_file,fn)

        
        # Load mapping from answers to labels
        self.ans2label = json.load(open(self.ans2label_file, 'rb'))[0]
        self.label2ans = {v:k for k,v in self.ans2label.items()}
        self.num_labels = len(self.label2ans)
        self.num_answers = len(self.ans2label)

        if os.path.exists(self.cached_data_file):
            # Load cached data
            self.data = pickle.load(open(self.cached_data_file,'rb'))
        else:
            self.annotations_file = os.path.join(data_dir, 'v2_mscoco_{}2014_annotations.json'.format(split))
            self.question_file = os.path.join(data_dir,'v2_OpenEnded_mscoco_{}2014_questions.json'.format(split))
            
            # Create map from question id to question
            questions = json.load(open(self.question_file))['questions']
            qid2qdata = {x['question_id']:x for x in questions}
            
            # Create data for each annotation
            annotations = json.load(open(self.annotations_file))['annotations']
            self.data = []
            for anno in annotations:
                qid = anno["question_id"]
                correct_answer = anno['multiple_choice_answer']
                image_id = anno["image_id"]
                
                # Retrieve the question for this annotation
                qdata = qid2qdata[qid]
                assert qdata["image_id"] == image_id
                question = qdata["question"]
                    
                # Map from each crowdsourced answer to occurences in annotation
                answers = [a['answer'] for a in anno['answers']]
                answer_count = defaultdict(int)
                
                for ans in answers:
                    answer_count[ans] += 1
                
                labels = []
                scores = []
                answers = []
                for answer in answer_count:
                    if answer not in self.ans2label:
                        continue
                    labels.append(self.ans2label[answer])
                    score = get_score(answer_count[answer])
                    scores.append(score)
                    answers.append(answer)
                
                example = {"question_id":qid,
                           "image_id":image_id,
                           "question":question,
                           "correct_answer":correct_answer,
                           "labels":labels,
                           "answers":answers,
                           "scores":scores}
                self.data.append(example)
            pickle.dump(self.data,open(self.cached_data_file,'wb'))
            
        self.n_examples = len(self.data)
        
    def __len__(self):
        return self.n_examples
    
    def __getitem__(self,
                    index:int) -> Dict:
        """

        Args:
            index (int): _description_
        """
        example = self.data[index]
        question_id = example['question_id']
        
        question = example['question']
        image_id = example["image_id"]
        image = self.get_pil_image(image_id)
        image = self.image_transform(image)
        
        labels = example['labels']
        scores = example['scores']
        target_scores = target_tensor(self.num_labels, labels, scores)
        
        encodings = self.processor(images=image,
                                   text=question,
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
        
        return {
            "encodings":encodings,
                "labels":labels,
                "target_scores":target_scores,
                "question_ids":question_id}
    def get_pil_image(self, image_id:str) -> Image:
        """Load image data corresponding to image_id, re-sizes and returns PIL.Image object

        Args:
            image_id (str): _description_

        Returns:
            Image: _description_
        """
        assert image_id in self.imageid2filename.keys()
        image_fn = self.imageid2filename[image_id]
        image = Image.open(image_fn)
        return image
    def vqa_batch_collate_ViLT(self,
                               batch:List[Dict],
                                ):
        """Collates each model input for all batch items into a single model input.

        Args:
            batch (List[Dict]): _description_
        """
        
        batch_labels = [x['labels'] for x in batch]
        batch_scores = [x['target_scores'] for x in batch]
        batch_scores = torch.stack(batch_scores,dim=0)
        
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
        
        return {"encodings":encodings,
                "labels":batch_scores}



def get_score(occurences):
    if occurences == 0:
        return 0.0
    elif occurences == 1:
        return 0.3
    elif occurences == 2:
        return 0.6
    elif occurences == 3:
        return 0.9
    else:
        return 1.0
    
    
def target_tensor(num_labels, labels, scores):
    """Create the taregt by labels and scores.

    Args:
        num_labels (_type_): _description_
        labels (_type_): _description_
        scores (_type_): _description_
    """
    target = torch.zeros(num_labels)
    target[labels] = torch.tensor(scores)
    return target