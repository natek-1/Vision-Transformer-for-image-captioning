from collections import defaultdict
import logging
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import tqdm
from tqdm import trange

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from transformers import AutoTokenizer

from vision.dataset.dataset import FlickrDataset
from vision.train.train import train_epoch, val_epoch 
from vision.model.caption import VisionEncoderDecoder, TokenDrop
from vision.utils import train_val_split, custom_collate_fn, save_checkpoint, create_visualizations

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("training.log")])
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.backends.cuda.matmul.allow_tf32 = True

# Define hyperparameters
LEARNING_RATE = 1e-4
IMAGE_SIZE = 128
NEPOCHS = 130
BATCH_SIZE = 128
HIDDEN_SIZE = 256
NUM_LAYERS= (6, 6)
NUM_HEADS = 8
PATCH_SIZE = 8

# Resume training configuration
RESUME_TRAINING = True  # Set to True to resume from checkpoint
CHECKPOINT_PATH = "checkpoints/model_checkpoint.pt"
CHECKPOINT_DIR = "checkpoints"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

folder = "flickr30k/Images/"
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


df = pd.read_csv("flickr30k/captions.txt")
caption_dict = defaultdict(list)
for _, row in df.iterrows():
    caption_dict[row.image].append(row.caption)
train_data, val_data = train_val_split(caption_dict)


train_transform = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                      transforms.RandomCrop(IMAGE_SIZE),
                                      #transforms.RandomHorizontalFlip(0.1),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225]),
                                      transforms.RandomErasing(p=0.5)]) 

val_transform = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                transforms.CenterCrop(IMAGE_SIZE),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])]) 

train_dataset = FlickrDataset(root_dir=folder, data_dict=train_data,
                        transform=train_transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=custom_collate_fn, num_workers=12, prefetch_factor=64, pin_memory=True)
val_dataset = FlickrDataset(root_dir=folder, data_dict=val_data,
                        transform=val_transform)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn,
                          num_workers=8, prefetch_factor=64, pin_memory=True)


model = VisionEncoderDecoder(image_size=IMAGE_SIZE, channels_in=3, num_emb=tokenizer.vocab_size,
                             patch_size=PATCH_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
                             num_heads=NUM_HEADS, mlp_dropout=0.1, att_dropout=0.1)
model = model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
td = TokenDrop(0.5)


num_model_params = 0
for param in model.parameters():
    num_model_params += param.flatten().shape[0]

print("This Model Has %d (Approximately %d Million) Parameters!" % (num_model_params, num_model_params//1e6))

# Initialize training metrics
train_losses = []
val_losses = []
meteor_scores = []
bleu_scores = {'bleu_1': [], 'bleu_2': [], 'bleu_3': [], 'bleu_4': []}
train_epoch_loss = []
val_epoch_loss = []
best_val_loss = float('inf')
start_epoch = 0

# Create checkpoint directory if it doesn't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Resume training if checkpoint exists and RESUME_TRAINING is True
if RESUME_TRAINING and os.path.exists(CHECKPOINT_PATH):
    print(f"Loading checkpoint from {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    
    # Load metric histories
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    meteor_scores = checkpoint['meteor_scores']
    bleu_scores = checkpoint['bleu_scores']
    train_epoch_loss = checkpoint['train_epoch_loss']
    val_epoch_loss = checkpoint['val_epoch_loss']
    
    print(f"Resumed from epoch {start_epoch} with best validation loss: {best_val_loss:.4f}")
    logging.info(f"Resumed training from epoch {start_epoch} with best validation loss: {best_val_loss:.4f}")
else:
    print("Starting training from scratch")
    logging.info("Starting training from scratch")


# Training loop
epoch_range = trange(start_epoch, NEPOCHS, leave=False, desc="Epoch")
for epoch in epoch_range:
    
    losses = train_epoch(model, train_loader, optimizer, DEVICE, loss_fn, td)
    train_losses.extend(losses)
    val_loss_list, bleu_1, bleu_2, bleu_3, bleu_4, meteor = val_epoch(model, val_loader, DEVICE,
                                                                   loss_fn, epoch, tokenizer, num_examples=3)
    val_losses.extend(val_loss_list)
    meteor_scores.append(meteor)
    bleu_scores['bleu_1'].append(bleu_1)
    bleu_scores['bleu_2'].append(bleu_2)
    bleu_scores['bleu_3'].append(bleu_3)
    bleu_scores['bleu_4'].append(bleu_4)
    train_epoch_loss.append(np.mean(losses))
    val_epoch_loss.append(np.mean(val_loss_list))
    
    epoch_range.set_postfix(train_loss=np.mean(losses), val_loss=np.mean(val_loss_list),
                            bleu_1=bleu_1, bleu_2=bleu_2, bleu_3=bleu_3, bleu_4=bleu_4, meteor=meteor)
    
    # Save checkpoint every epoch
    save_checkpoint(epoch, model, optimizer, best_val_loss,
                   train_losses, val_losses, meteor_scores, bleu_scores,
                   train_epoch_loss, val_epoch_loss, CHECKPOINT_PATH)
    
    # Save best model weights
    if np.mean(val_loss_list) < best_val_loss:
        best_val_loss = np.mean(val_loss_list)
        torch.save(model.state_dict(), "model_weights.pt")
        logging.info(f"\n--- Model weights saved at Epoch {epoch} with Validation Loss: {best_val_loss:.4f} ---\n")


# Create visualizations at the end of training
print("\nGenerating visualizations...")
create_visualizations(train_losses, val_losses, train_epoch_loss, val_epoch_loss,
                     bleu_scores, meteor_scores, start_epoch)
print("Training complete!")