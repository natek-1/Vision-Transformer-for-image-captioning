import os
import logging

import numpy as np

import torch


import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("training.log")])


def train_val_split(caption_data, train_size=0.8, shuffle=True):
    """Split the captioning dataset into train and validation sets.

    Args:
        caption_data (dict): Dictionary containing the mapped caption data
        train_size (float): Fraction of all the full dataset to use as training data
        shuffle (bool): Whether to shuffle the dataset before splitting

    Returns:
        Traning and validation datasets as two separated dicts
    """

    all_images = list(caption_data.keys())

    if shuffle:
        np.random.shuffle(all_images)

    train_size = int(len(caption_data) * train_size)

    training_data = {
        img_name: caption_data[img_name] for img_name in all_images[:train_size]
    }
    validation_data = {
        img_name: caption_data[img_name] for img_name in all_images[train_size:]
    }
    return training_data, validation_data


def custom_collate_fn(batch):
    '''
    Custom collate function to handle batching of images and captions.
    Args:
        batch (list): List of tuples where each tuple is (image, caption_dict)
    Returns:
        dict: A dictionary with batched 'pixel_values', 'input_ids', and 'attention
    '''
    images = []
    captions = []
    attention_mask = []
    
    for entry in batch:
        images.append(entry[0])
        captions.append(entry[1]['input_ids'])
        attention_mask.append(entry[1]['attention_mask'])
        
    # Stack images into a 4D tensor [B, C, H, W]
    images = torch.stack(images, dim=0)
    
    captions = torch.stack(captions, dim=0)
    
    return {
        'pixel_values': images,
        'input_ids': captions,
        'attention_mask': torch.stack(attention_mask, dim=0)
    }
    
def save_checkpoint(epoch, model, optimizer, best_val_loss, 
                   train_losses, val_losses, meteor_scores, bleu_scores,
                   train_epoch_loss, val_epoch_loss, checkpoint_path):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'meteor_scores': meteor_scores,
        'bleu_scores': bleu_scores,
        'train_epoch_loss': train_epoch_loss,
        'val_epoch_loss': val_epoch_loss
    }
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Checkpoint saved at epoch {epoch}")


def create_visualizations(train_losses, val_losses, train_epoch_loss, val_epoch_loss,
                         bleu_scores, meteor_scores, start_epoch):
    """Create both static and interactive visualizations of training metrics"""
    
    # Create output directory for plots
    os.makedirs("plots", exist_ok=True)
    
    # 1. Static plots using matplotlib
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Metrics', fontsize=16, fontweight='bold')
    
    # Plot 1: Training loss (step-wise)
    axes[0, 0].plot(train_losses, alpha=0.6, linewidth=0.5)
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss (Step-wise)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Validation loss (step-wise)
    axes[0, 1].plot(val_losses, alpha=0.6, linewidth=0.5, color='orange')
    axes[0, 1].set_xlabel('Validation Step')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Validation Loss (Step-wise)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Train and Val loss per epoch
    epochs = range(start_epoch, start_epoch + len(train_epoch_loss))
    axes[1, 0].plot(epochs, train_epoch_loss, marker='o', label='Train Loss', linewidth=2)
    axes[1, 0].plot(epochs, val_epoch_loss, marker='s', label='Val Loss', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Train and Validation Loss (Epoch-wise)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: BLEU scores and METEOR
    axes[1, 1].plot(epochs, bleu_scores['bleu_1'], marker='o', label='BLEU-1', linewidth=2)
    axes[1, 1].plot(epochs, bleu_scores['bleu_2'], marker='s', label='BLEU-2', linewidth=2)
    axes[1, 1].plot(epochs, bleu_scores['bleu_3'], marker='^', label='BLEU-3', linewidth=2)
    axes[1, 1].plot(epochs, bleu_scores['bleu_4'], marker='d', label='BLEU-4', linewidth=2)
    axes[1, 1].plot(epochs, meteor_scores, marker='*', label='METEOR', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('BLEU Scores and METEOR (Epoch-wise)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/training_metrics_static.png', dpi=300, bbox_inches='tight')
    print("Static plots saved to 'plots/training_metrics_static.png'")
    plt.close()
    
    # 2. Interactive plots using plotly
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Training Loss (Step-wise)', 'Validation Loss (Step-wise)',
                       'Train and Validation Loss (Epoch-wise)', 'BLEU Scores and METEOR (Epoch-wise)'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Plot 1: Training loss (step-wise)
    fig.add_trace(
        go.Scatter(y=train_losses, mode='lines', name='Train Loss (steps)',
                  line=dict(width=1), opacity=0.7),
        row=1, col=1
    )
    
    # Plot 2: Validation loss (step-wise)
    fig.add_trace(
        go.Scatter(y=val_losses, mode='lines', name='Val Loss (steps)',
                  line=dict(width=1, color='orange'), opacity=0.7),
        row=1, col=2
    )
    
    # Plot 3: Train and Val loss per epoch
    fig.add_trace(
        go.Scatter(x=list(epochs), y=train_epoch_loss, mode='lines+markers',
                  name='Train Loss (epoch)', line=dict(width=2)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=list(epochs), y=val_epoch_loss, mode='lines+markers',
                  name='Val Loss (epoch)', line=dict(width=2)),
        row=2, col=1
    )
    
    # Plot 4: BLEU scores and METEOR
    fig.add_trace(
        go.Scatter(x=list(epochs), y=bleu_scores['bleu_1'], mode='lines+markers',
                  name='BLEU-1', line=dict(width=2)),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=list(epochs), y=bleu_scores['bleu_2'], mode='lines+markers',
                  name='BLEU-2', line=dict(width=2)),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=list(epochs), y=bleu_scores['bleu_3'], mode='lines+markers',
                  name='BLEU-3', line=dict(width=2)),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=list(epochs), y=bleu_scores['bleu_4'], mode='lines+markers',
                  name='BLEU-4', line=dict(width=2)),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=list(epochs), y=meteor_scores, mode='lines+markers',
                  name='METEOR', line=dict(width=2)),
        row=2, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text="Training Step", row=1, col=1)
    fig.update_xaxes(title_text="Validation Step", row=1, col=2)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=2)
    
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=2, col=1)
    fig.update_yaxes(title_text="Score", row=2, col=2)
    
    fig.update_layout(
        title_text="Training Metrics - Interactive Dashboard",
        height=900,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.write_html('plots/training_metrics_interactive.html')
    print("Interactive plots saved to 'plots/training_metrics_interactive.html'")