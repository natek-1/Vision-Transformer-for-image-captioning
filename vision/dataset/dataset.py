from typing import List
import random

import torch
import torch.nn as nn

from vision.model.tokenizer import TOKENIZER



class SampleCaption(nn.Module):
    def __init__(self, training=True):
        self.training = training
    def __call__(self, sample):
        rand_index = random.randint(0, len(sample) - 1)
        #print(sample)
        return sample[rand_index] if self.training else sample[:5]
    


def custom_collate_fn(batch: List, tokenizer = TOKENIZER, train: bool =True):
    images = []
    captions = []

    for entry in batch:
        images.append(entry[0])
        captions.append(entry[1])

    sample_captions = captions if train else [caption[0] for caption in captions]
    token_ids = tokenizer(sample_captions, padding=True, return_tensors='pt')  
    images = torch.stack(images, dim=0)
    return images, token_ids['input_ids'], token_ids['attention_mask'], captions