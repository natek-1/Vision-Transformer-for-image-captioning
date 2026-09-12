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
import torchvision.datasets as datasets

from transformers import ViTImageProcessor

from vision.dataset.dataset import SampleCaption, custom_collate_fn
from vision.train.train import train_epoch, val_epoch 
from vision.model.caption import VisionEncoderDecoder, TokenDrop
from vision.model.tokenizer import TOKENIZER
from vision.utils import save_checkpoint, create_visualizations, set_seed, get_scheduler

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("training.log")])
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.backends.cuda.matmul.allow_tf32 = True

# Define hyperparameters
LEARNING_RATE = 1e-4 #2e-4 # after current experiment try to start with 6e-4
IMAGE_SIZE = 224
NEPOCHS = 50
BATCH_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 12
NUM_HEADS = 8

# Learning rate scheduler configuration
SCHEDULER_TYPE = None  # Options: "cosine", "cosine_warmup", "step", "plateau", "onecycle", "noam", None
WARMUP_EPOCHS = 5
MIN_LR = 2e-5 # 6e-6 # after current experiment try to start with 6e-5

# Resume training configuration
RESUME_TRAINING = False  # Set to True to resume from checkpoint
CHECKPOINT_PATH = "checkpoints/model_checkpoint.pt"
CHECKPOINT_DIR = "checkpoints"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


PROCESSOR = ViTImageProcessor.from_pretrained('google/vit-large-patch16-224-in21k')
tokenizer = TOKENIZER
set_seed(42)

import os, shutil, pathlib
  
def stage_to_local(src_dir, local_root="/tmp", subdir="coco_cache"):
    """Copy an image directory to fast node-local disk once; return the local path.
    Falls back to the original NFS path if local staging isn't possible."""
    src = pathlib.Path(src_dir).resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"source dir not found: {src}")

    # unique per-user destination on local disk
    dest_base = pathlib.Path(local_root) / f"{os.environ.get('USER','user')}_{subdir}"
    dest = dest_base / src.name

    try:
        if dest.is_dir():
            # already staged if counts match; otherwise re-copy
            n_src = sum(1 for _ in src.iterdir())
            n_dst = sum(1 for _ in dest.iterdir())
            if n_dst >= n_src:
                print(f"[stage] using cached {dest} ({n_dst} entries)")
                return str(dest)
            print(f"[stage] incomplete cache ({n_dst}/{n_src}), re-copying")
            shutil.rmtree(dest, ignore_errors=True)

        dest_base.mkdir(parents=True, exist_ok=True)
        print(f"[stage] copying {src} -> {dest} (one-time)...")
        shutil.copytree(src, dest)
        print(f"[stage] done: {dest}")
        return str(dest)
    except OSError as e:
        # e.g. out of space on /tmp — fall back to NFS
        print(f"[stage] WARNING: local staging failed ({e}); using NFS path {src}")
        return str(src)
  
# Usage in the notebook, before creating the datasets:
COCO_ROOT = "/work/ngkuissi/Vision-Transformer-for-image-captioning/datasets"
train_dir = stage_to_local(os.path.join(COCO_ROOT, "train2014"))
val_dir   = stage_to_local(os.path.join(COCO_ROOT, "val2014"))

data_set_root='datasets'
train_set ='train2014'
validation_set ='val2014'

train_image_path = os.path.join(data_set_root, train_set)
train_ann_file = '{}/annotations/captions_{}.json'.format(data_set_root, train_set)

val_image_path = os.path.join(data_set_root, validation_set)
val_ann_file = '{}/annotations/captions_{}.json'.format(data_set_root, validation_set)

train_image_path = stage_to_local(os.path.join(data_set_root, train_set))
val_image_path   = stage_to_local(os.path.join(data_set_root, validation_set))


train_dataset = datasets.CocoCaptions(root=train_image_path,
                                      annFile=train_ann_file,
                                      transform=lambda x: PROCESSOR(images=x, return_tensors="pt")["pixel_values"].squeeze(0),
                                      target_transform=SampleCaption())

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True,
                          collate_fn=custom_collate_fn, num_workers=14, prefetch_factor=30)


val_dataset = datasets.CocoCaptions(root=val_image_path,
                                    annFile=val_ann_file,
                                    transform=lambda x: PROCESSOR(images=x, return_tensors="pt")["pixel_values"].squeeze(0),
                                    target_transform=SampleCaption(False))
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: custom_collate_fn(x, train=False),
                          num_workers=14, prefetch_factor=30)


model = VisionEncoderDecoder(vocab_size=tokenizer.vocab_size, max_length=70, 
                            num_layers=NUM_LAYERS, hidden_size=HIDDEN_SIZE, 
                            num_heads=NUM_HEADS)
model = model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                              weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, label_smoothing=0.1,
                                    reduction="none")
td = TokenDrop(0.5)

num_training_steps = len(train_loader) * NEPOCHS
scheduler, is_batch_level_scheduler = get_scheduler(optimizer, SCHEDULER_TYPE, NEPOCHS, num_training_steps,
                                                    min_lr=MIN_LR, learning_rate=LEARNING_RATE, warmup_epochs=WARMUP_EPOCHS,
                                                    hidden_size=HIDDEN_SIZE)
scaler = torch.cuda.amp.GradScaler()


num_model_params = 0
trainable = 0
for param in model.parameters():
    num_model_params += param.flatten().shape[0]
    trainable += param.flatten().shape[0] if param.requires_grad else 0

print(f"-This Model Has {num_model_params} (Approximately {num_model_params//1e6:.2f} Million) Parameters!")
print(f"This Model Has {trainable} (Approximately {trainable/1e6:.2f} Million) Trainable Parameters!")
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
                            scheduler=scheduler if is_batch_level_scheduler else None, scaler=scaler)
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