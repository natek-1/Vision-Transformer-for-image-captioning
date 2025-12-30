import os
from PIL import Image

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer

from vision.tokenizer.vocab import Vocabulary




class FlickrDataset(Dataset):
    max_len = 90
    def __init__(self, root_dir, data_dict, transform=None):
        self.root_dir = root_dir
        self.data_dict = data_dict
        self.transform = transform

        # get the image and caption
        self.item = self.setup_item()

        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    def __len__(self):
        return len(self.item)
    
    def setup_item(self):
        item = []
        for image_id, image_captions in self.data_dict.items():
            item.append((image_id, image_captions))
        return item


    def __getitem__(self, index):
        # get image
        image_path = os.path.join(self.root_dir, self.item[index][0]) # path to image
        img = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        # get caption
        caption = self.item[index][1] # list of relevant caption
        num_caption = self.tokenizer(caption, add_special_tokens=True, max_length=FlickrDataset.max_len, padding='max_length', truncation=True,
                                    return_tensors="pt")
        return img, num_caption
