import os
from PIL import Image

import torch
from torch.utils.data import Dataset

from vision.tokenizer.vocab import Vocabulary


class FlickrDataset(Dataset):
    max_len = 45
    def __init__(self, root_dir, data_dict, vocabulary: Vocabulary, transform=None, train=True):
        self.root_dir = root_dir
        self.data_dict = data_dict
        self.transform = transform

        # get the image and caption
        self.train = train
        self.caption = []
        self.item = self.setup_item()

        # Create our own vocabulary
        self.vocabulary = vocabulary
        self.eos_token = vocabulary.tokenizer.eos_token
        self.pad_token = vocabulary.tokenizer.pad_token
    
    def __len__(self):
        return len(self.item)
    
    def setup_item(self):
        item = []
        if self.train:
            for image_id, image_captions in self.data_dict.items():
                for caption in image_captions:
                    item.append((image_id, caption))
        else:
            for image_id, image_captions in self.data_dict.items():
                item.append((image_id, image_captions))
        return item


    def __getitem__(self, index):
        # get image
        image_path = os.path.join(self.root_dir, self.item[index][0])
        img = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        # get caption
        caption = self.item[index][1]
        
        if self.train:
            caption += self.eos_token
            caption = self.vocabulary.encode(caption, True, True)
            return img, torch.tensor(caption.input_ids, dtype=torch.long), torch.tensor(caption.attention_mask, dtype=torch.long)
        else:
            captions = torch.zeros(5, self.vocabulary.max_len).to(torch.long)
            atten_mask = torch.zeros(5, self.vocabulary.max_len).to(torch.long)
            for idx, cap in enumerate(caption):
                cap += self.eos_token
                cap = self.vocabulary.encode(cap,True, True)
                captions[idx] = torch.tensor(cap.input_ids, dtype=torch.long)
                atten_mask[idx] = torch.tensor(cap.attention_mask, dtype=torch.long)
            return img, captions, atten_mask