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
LEARNING_RATE = 6e-4 #2e-4 # after current experiment try to start with 6e-4
IMAGE_SIZE = 128
NEPOCHS = 130
BATCH_SIZE = 128
HIDDEN_SIZE = 256
NUM_LAYERS = (6, 6)
NUM_HEADS = 8
PATCH_SIZE = 8

# Learning rate scheduler configuration
SCHEDULER_TYPE = "cosine"  # Options: "cosine", "cosine_warmup", "step", "plateau", "onecycle", "noam", None
WARMUP_EPOCHS = 5
MIN_LR = 2e-5 # 6e-6 # after current experiment try to start with 6e-5

# Resume training configuration
RESUME_TRAINING = False  # Set to True to resume from checkpoint
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
                                      transforms.RandomHorizontalFlip(0.3),
                                      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225]),
                                      transforms.RandomErasing(p=0.1)])

val_transform = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                transforms.CenterCrop(IMAGE_SIZE),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])]) 

train_dataset = FlickrDataset(root_dir=folder, data_dict=train_data,
                        transform=train_transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=custom_collate_fn, num_workers=12, prefetch_factor=24, 
                          pin_memory=True, persistent_workers=True)
val_dataset = FlickrDataset(root_dir=folder, data_dict=val_data,
                        transform=val_transform)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn,
                          num_workers=8, prefetch_factor=24, pin_memory=True, persistent_workers=True)


model = VisionEncoderDecoder(image_size=IMAGE_SIZE, channels_in=3, num_emb=tokenizer.vocab_size,
                             patch_size=PATCH_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
                             num_heads=NUM_HEADS, mlp_dropout=0.3, att_dropout=0.3)
model = model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4,
                              betas=(0.9, 0.999), eps=1e-8)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, label_smoothing=0.1)  # Added label_smoothing
td = TokenDrop(0.5)


# Setup learning rate scheduler
def get_scheduler(optimizer, scheduler_type, num_epochs, num_training_steps):
    """Create learning rate scheduler based on specified type.
    
    Returns:
        tuple: (scheduler, is_batch_level) where is_batch_level indicates if scheduler
               should be stepped per batch (True) or per epoch (False)
    """
    if scheduler_type == "cosine":
        # Epoch-level scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=MIN_LR
        )
        return scheduler, False
        
    elif scheduler_type == "cosine_warmup":
        # Batch-level scheduler with warmup
        warmup_steps = WARMUP_EPOCHS * num_training_steps // num_epochs
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            # Cosine annealing after warmup
            progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
            return max(MIN_LR / LEARNING_RATE, 0.5 * (1.0 + np.cos(np.pi * progress)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return scheduler, True
        
    elif scheduler_type == "noam":
        # Transformer/Noam scheduler (Attention is All You Need)
        warmup_steps = WARMUP_EPOCHS * num_training_steps // num_epochs
        
        def lr_lambda(current_step):
            current_step = max(1, current_step)  # Avoid division by zero
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(warmup_steps)
            # Inverse square root decay
            return float(warmup_steps) ** 0.5 / float(current_step) ** 0.5
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return scheduler, True
        
    elif scheduler_type == "step":
        # Epoch-level scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.5
        )
        return scheduler, False
        
    elif scheduler_type == "plateau":
        # Epoch-level scheduler (validation-based)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=MIN_LR
        )
        return scheduler, False
        
    elif scheduler_type == "onecycle":
        # Batch-level scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=LEARNING_RATE, total_steps=num_training_steps,
            pct_start=0.3, anneal_strategy='cos'
        )
        return scheduler, True
    else:
        return None, False

num_training_steps = len(train_loader) * NEPOCHS
scheduler, is_batch_level_scheduler = get_scheduler(optimizer, SCHEDULER_TYPE, NEPOCHS, num_training_steps)


num_model_params = 0
for param in model.parameters():
    num_model_params += param.flatten().shape[0]

print("This Model Has %d (Approximately %d Million) Parameters!" % (num_model_params, num_model_params//1e6))
if scheduler is not None:
    print(f"Using {SCHEDULER_TYPE} scheduler ({'batch-level' if is_batch_level_scheduler else 'epoch-level'})")

# Initialize training metrics
train_losses = []
val_losses = []
meteor_scores = []
bleu_scores = {'bleu_1': [], 'bleu_2': [], 'bleu_3': [], 'bleu_4': []}
train_epoch_loss = []
val_epoch_loss = []
learning_rates = []
best_val_loss = float('inf')
best_meteor = 0.0
start_epoch = 0

# Create checkpoint directory if it doesn't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Resume training if checkpoint exists and RESUME_TRAINING is True
if RESUME_TRAINING and os.path.exists(CHECKPOINT_PATH):
    print(f"Loading checkpoint from {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    best_meteor = checkpoint.get('best_meteor', 0.0)
    
    # Load metric histories
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    meteor_scores = checkpoint.get('meteor_scores', [])
    bleu_scores = checkpoint.get('bleu_scores', {'bleu_1': [], 'bleu_2': [], 'bleu_3': [], 'bleu_4': []})
    train_epoch_loss = checkpoint.get('train_epoch_loss', [])
    val_epoch_loss = checkpoint.get('val_epoch_loss', [])
    learning_rates = checkpoint.get('learning_rates', [])
    
    print(f"Resumed from epoch {start_epoch} | Best Val Loss: {best_val_loss:.4f} | Best METEOR: {best_meteor:.4f}")
    logging.info(f"Resumed training from epoch {start_epoch} | Best Val Loss: {best_val_loss:.4f} | Best METEOR: {best_meteor:.4f}")
else:
    print("Starting training from scratch")
    logging.info("Starting training from scratch")


# Training loop with keyboard interrupt handling
try:
    epoch_range = trange(start_epoch, NEPOCHS, leave=False, desc="Epoch")
    for epoch in epoch_range:
        
        # Pass scheduler to train_epoch only if it's batch-level
        losses = train_epoch(model, train_loader, optimizer, DEVICE, loss_fn, td, 
                            scheduler if is_batch_level_scheduler else None)
        train_losses.extend(losses)
        val_loss_list, bleu_1, bleu_2, bleu_3, bleu_4, meteor = val_epoch(model, val_loader, DEVICE,
                                                                       loss_fn, epoch, tokenizer, num_examples=3)
        val_losses.extend(val_loss_list)
        meteor_scores.append(meteor)
        bleu_scores['bleu_1'].append(bleu_1)
        bleu_scores['bleu_2'].append(bleu_2)
        bleu_scores['bleu_3'].append(bleu_3)
        bleu_scores['bleu_4'].append(bleu_4)
        
        avg_train_loss = np.mean(losses)
        avg_val_loss = np.mean(val_loss_list)
        train_epoch_loss.append(avg_train_loss)
        val_epoch_loss.append(avg_val_loss)
        
        # Track learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # Step scheduler (if not batch-level scheduler)
        if scheduler is not None and not is_batch_level_scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
        
        epoch_range.set_postfix(train_loss=avg_train_loss, val_loss=avg_val_loss,
                                bleu_1=bleu_1, bleu_2=bleu_2, bleu_3=bleu_3, bleu_4=bleu_4, 
                                meteor=meteor, lr=current_lr)
        
        # Enhanced logging
        logging.info(
            f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"METEOR: {meteor:.4f} | BLEU-4: {bleu_4:.4f} | LR: {current_lr:.6f}"
        )
        
        # Save checkpoint every epoch with scheduler state
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_val_loss': best_val_loss,
            'best_meteor': best_meteor,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'meteor_scores': meteor_scores,
            'bleu_scores': bleu_scores,
            'train_epoch_loss': train_epoch_loss,
            'val_epoch_loss': val_epoch_loss,
            'learning_rates': learning_rates
        }
        torch.save(checkpoint_data, CHECKPOINT_PATH)
        
        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_val_loss_model.pt"))
            logging.info(f"✓ New best validation loss: {best_val_loss:.4f}")
        
        # Save best model based on METEOR score
        if meteor > best_meteor:
            best_meteor = meteor
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_meteor_model.pt"))
            logging.info(f"✓ New best METEOR score: {best_meteor:.4f}")

except KeyboardInterrupt:
    print("\n\n" + "="*80)
    print("Training interrupted by user (Ctrl+C)")
    print("="*80)
    logging.info(f"Training interrupted by user at epoch {epoch}")
finally:
    # Always generate visualizations, whether training completed or was interrupted
    print("\nGenerating visualizations...")
    try:
        create_visualizations(train_losses, val_losses, train_epoch_loss, val_epoch_loss,
                             bleu_scores, meteor_scores, start_epoch)
        print("Visualizations generated successfully!")
    except Exception as e:
        print(f"Warning: Could not generate visualizations: {e}")
        logging.warning(f"Could not generate visualizations: {e}")
    
    print("\n" + "="*80)
    print("Training Summary:")
    print(f"  Epochs completed: {len(train_epoch_loss)}")
    print(f"  Best Validation Loss: {best_val_loss:.4f}")
    print(f"  Best METEOR Score: {best_meteor:.4f}")
    print("="*80)