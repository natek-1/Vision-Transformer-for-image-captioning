import logging
import random

from tqdm import tqdm
import numpy as np

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score

import torch

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("training.log")])

def calculate_corpus_meteor(references, hypotheses):
    '''
    Calculate the corpus-level METEOR score.
    Args:
        references (list): List of lists of reference sentences
        hypotheses (list): List of predicted sentences
    Returns:
        float: Corpus-level METEOR score
    '''
    assert len(hypotheses) == len(references), "The number of predicted and expected sentences must be the same."
    scores = []
    for hyp, refs in zip(hypotheses, references):
        hyp_tokens = hyp.lower().split()
        ref_tokens_list = [ref.lower().split() for ref in refs]
        score = meteor_score(ref_tokens_list, hyp_tokens)
        scores.append(score)
    meteor_score_res = np.mean(scores)
    return meteor_score_res

def calculate_corpus_bleu(references, hypotheses, weight=(0.25, 0.25, 0.25, 0.25)):
    """Calculate the corpus-level BLEU score.

    Args:
        references (list): List of lists of reference sentences
        hypotheses (list): List of predicted sentences
        weight (tuple): Weights for n-gram precision
    Returns:
        float: Corpus-level BLEU score
    """
    assert len(hypotheses) == len(references), "The number of predicted and expected sentences must be the same."
    #shape: [n_samples, n_tokens]
    hyp_tokens = [hyp.lower().split() for hyp in hypotheses]
    #shape: [n_samples, 5, n_tokens]
    ref_tokens = [[ref.lower().split() for ref in refs] for refs in references]
    return corpus_bleu(ref_tokens, hyp_tokens, weights=weight)


def train_epoch(caption_model, train_loader, optimizer,
                device, loss_fn, td, scaler, scheduler=None):
    caption_model.train()
    epoch_loss = []
    batch_iterator = tqdm(train_loader, desc="Training", leave=False)
    for batch in batch_iterator:
        images, token_ids, attention_mask, _ = batch
        images = images.to(device)
        token_ids = token_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Shift the captions to create input and target sequences
        input_captions = token_ids[:,:-1]
        target_captions = token_ids[:, 1:]

        # forward pass
        input_captions = td(input_captions)
        outputs = caption_model(input_image=images, 
                                input_seq=input_captions, 
                                padding_mask=attention_mask[:,:-1])
        # Compute the loss
        loss_mask = (~(target_captions == 0)).float()
        loss = loss_fn(outputs.transpose(1, 2), 
                        target_captions)
        loss = (loss * loss_mask).sum() / loss_mask.sum()
        
        # Backpropagation and optimization step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        # Gradient clipping to prevent exploding gradients
        #torch.nn.utils.clip_grad_norm_(caption_model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Step scheduler if it's a batch-level scheduler (OneCycleLR)
        if scheduler is not None and hasattr(scheduler, 'step') and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()
        
        epoch_loss.append(loss.item())
        batch_iterator.set_postfix(loss=loss.item())
    
    return epoch_loss


def val_epoch(model, val_loader, device, loss_fn,
              epoch, tokenizer, num_examples=5):
    epoch_loss = []
    references = []
    hypotheses = []
    model.eval()
    with torch.inference_mode():
        batch_iterator = tqdm(enumerate(val_loader), total=len(val_loader),
                             desc=f"Evaluation Processing Epoch: {epoch:02d}", leave=False)
        for idx, batch in batch_iterator:
            images, token_ids, attention_mask, full_captions  = batch
            images = images.to(device)
            token_ids = token_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Shift the target sequence to the right by one position
            sample_caption = token_ids[:,:-1].clone()
            sample_att_mask = attention_mask [:,:-1].clone()
            target_captions = token_ids[:,1:].clone()

            # Forward pass through the model
            outputs = model(input_image=images, input_seq=sample_caption, 
                            padding_mask=sample_att_mask)

            # Compute the loss
            loss_mask = (~(target_captions == 0)).float()
            loss = loss_fn(outputs.transpose(1, 2), 
                        target_captions)
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            epoch_loss.append(loss.item())
            batch_iterator.set_postfix(loss=loss.item())
            
            # raw predictions
            if epoch > 20:
                predictions = model.generate(input_image=images, 
                                    sos_token=tokenizer.cls_token_id,
                                    eos_token=tokenizer.sep_token_id, 
                                    pad_token=tokenizer.pad_token_id, 
                                    max_length=70, temp=0.1)
                # Gather references and hypotheses for metric calculation
                hypotheses.extend(tokenizer.batch_decode(predictions, skip_special_tokens=True))
                references.extend(full_captions)

        #print samples
        
        if len(hypotheses) > 0:
            indices = random.sample(range(len(hypotheses)), num_examples)
            logging.info(f"\n--- Epoch {epoch}, Sample {num_examples} Predictions vs References ---\n")
            for idx in indices:
                logging.info("\nModel Prediction:")
                logging.info(hypotheses[idx])
                logging.info("\nData References:")
                for i in range(5):
                    logging.info(references[idx][i])
            logging.info(str("-" * 100))
        
        # Calculate BLEU and METEOR scores
        if epoch > 20:
            bleu_1 = calculate_corpus_bleu(references, hypotheses, weight=(1, 0, 0, 0))
            bleu_2 = calculate_corpus_bleu(references, hypotheses, weight=(0.5, 0.5, 0, 0))
            bleu_3 = calculate_corpus_bleu(references, hypotheses, weight=(0.33, 0.33, 0.33, 0))
            bleu_4 = calculate_corpus_bleu(references, hypotheses)
            meteor = calculate_corpus_meteor(references, hypotheses)
        else:
            bleu_1 = 0
            bleu_2 = 0
            bleu_3 = 0
            bleu_4 = 0
            meteor = 0

    return epoch_loss, bleu_1, bleu_2, bleu_3, bleu_4, meteor