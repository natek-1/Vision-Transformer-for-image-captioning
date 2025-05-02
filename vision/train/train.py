import random

from tqdm  import tqdm
import numpy as np

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score

import torch



def corpus_meteor(expected, predicted):
    meteor_score_sentences_list = list()
    [meteor_score_sentences_list.append(meteor_score(expect, predict)) for expect, predict in zip(expected, predicted)]
    meteor_score_res = np.mean(meteor_score_sentences_list)
    return meteor_score_res


# Function to compute loss while ignoring padding tokens
def compute_loss(logits, labels, mask, loss_fct):
    """
    Compute cross entropy loss while ignoring padding tokens.
    
    Args:
        logits: Output from model, shape [batch_size, seq_len, vocab_size]
        labels: Target token ids, shape [batch_size, seq_len]
        attention_mask: Mask indicating valid tokens, shape [batch_size, seq_len]
    
    Returns:
        Loss value
    """
    # shape: [batch_size * (seq_len)]
    losses = loss_fct(
        logits.view(-1, logits.size(-1)),
        labels.view(-1)
    )
    
    # Apply the mask to get losses only for non-padding tokens
    # shape: [batch_size * (seq_len)]
    masked_losses = losses * mask.view(-1)
    
    # Get the sum of losses and divide by number of non-padding tokens
    loss = masked_losses.sum() / mask.sum()
    
    return loss

def train_epoch(train_loader, model, device, criterion, optimizer, epoch, writer, global_step,
                teacher_force = False, teacher_forcing_ratio_start=0.8, teacher_forcing_ratio_end=0, epochs=5, vocab_size=50257):
    losses = []

    model.train()
    # Calculate teacher forcing ratio for this epoch (linearly decrease)
    teacher_forcing_ratio = teacher_forcing_ratio_start - (epoch / (epochs - 1)) * \
                            (teacher_forcing_ratio_start - teacher_forcing_ratio_end)
                            
    batch_iterator = tqdm(enumerate(train_loader),total=len(train_loader) , desc=f"Training Processing Epoch: {epoch:02d}", leave=False)
    for idx, (imgs, caps, att_mask) in batch_iterator:
        # move tensor to device if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        att_mask = att_mask.to(device)

        optimizer.zero_grad()

        # forward prop
        if teacher_force:
            predictions = model(imgs, caps[:,:-1], att_mask[:,:-1])
        else:
            encoder_output = model.vit(imgs).pooler_output # (batch_size, embed_dim)
            predictions = torch.zeros(caps.shape[0], caps.shape[1], vocab_size, device=imgs.device)
            start_output = torch.tensor([[]]*imgs.shape[0], device=imgs.device, dtype=torch.long) # batch_size, 1
            start_att = torch.tensor([[]]*imgs.shape[0], device=imgs.device, dtype=torch.long) # batch_size, 1
            
            for t in range(caps.shape[1]):
                logits = model.gpt2(start_output, encoder_output, start_att)[:,-1,:] # (batch_size, vocab_size)
                predictions[:, t] = logits.squeeze(1)
                use_teacher_forcing = random.random() < teacher_forcing_ratio
                
                if use_teacher_forcing:
                    start_output = torch.cat([start_output, caps[:, t].unsqueeze(1)], dim=1)
                    start_att = torch.cat([start_att, att_mask[:,t].unsqueeze(1)], dim=1)
                else:
                    new_token_mask = torch.ones((imgs.shape[0], 1), dtype=torch.long, device=imgs.device)
                    for i in range(imgs.shape[0]):
                        if logits[i].argmax(-1).item() == vocab_size-1:
                            new_token_mask[i, 0] = 0
                    start_output = torch.cat([start_output, logits.argmax(dim=-1, keepdim=True)], dim=1)
                    start_att = torch.cat([start_att, new_token_mask], dim=1)


        loss = compute_loss(predictions, caps, start_att, criterion)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
        optimizer.step()
        
        batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
        writer.add_scalar("Training loss", loss.item(), global_step=global_step)
        global_step += 1

        # keep track of metrics
        losses.append(loss.item())
        break
        

    print('Training Epoch #: [{0}]\t'
        'Loss: {loss:.4f}\t'.format(
                epoch, loss=np.mean(losses)))

    return global_step



def val_epoch(model, device, validation_loader, vocabulary, criterion, global_step, writer, epoch, num_examples=3):
    model.eval()
    epoch_loss = []
    references = []
    hypotheses = []
    alt_hyp = []
    with torch.inference_mode():
        batch_iterator = tqdm(enumerate(validation_loader),total=len(validation_loader) ,desc=f"Evaluation Processing Epoch: {epoch:02d}", leave=False)
        for idx, (images, captions, att_mask) in batch_iterator:
            images = images.to(device)
            captions = captions.to(device)
            att_mask = att_mask.to(device)

            # forward prop
            sample_caption = captions[:,0,:].clone()
            sample_att_mask = att_mask[:,0,:].clone()
            
            encoder_output = model.vit(images).pooler_output # (batch_size, embed_dim)
            predictions = torch.zeros(sample_caption.shape[0], sample_caption.shape[1], vocabulary.vocab_size(), device=images.device)
            start_output = torch.tensor([[]]*images.shape[0], device=images.device, dtype=torch.long) # batch_size, 1
            start_att = torch.tensor([[]]*images.shape[0], device=images.device, dtype=torch.long) # batch_size, 1
            
            for t in range(sample_caption.shape[1]):
                logits = model.gpt2(start_output, encoder_output, start_att)[:,-1,:] # (batch_size, vocab_size)
                predictions[:, t] = logits.squeeze(1)
                new_token_mask = torch.ones((images.shape[0], 1), dtype=torch.long, device=images.device)
                for i in range(images.shape[0]):
                    if logits[i].argmax(-1).item() == vocabulary.vocab_size()-1:
                        new_token_mask[i, 0] = 0
                start_output = torch.cat([start_output, logits.argmax(dim=-1, keepdim=True)], dim=1)
                start_att = torch.cat([start_att, new_token_mask], dim=1)

            loss = compute_loss(predictions, sample_caption, start_att, criterion)

            # keep track of metrics
            epoch_loss.append(loss.item())
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            writer.add_scalar("Validation loss", loss.item(), global_step=global_step)
            global_step += 1
            # gather all references and hypothesis for blue and meteor score calculations
            # references
            for i in range(captions.shape[0]): #looping over the batch dimension
                caps = []
                for j in range(captions.shape[1]):
                    caps.append(vocabulary.decode(captions[i,j].tolist()).split())
                references.append(caps)
            
            # Hypothesis
            predictions = predictions.argmax(dim=-1).tolsit()
            for prediction in predictions:
                hypotheses.append(vocabulary.decode(prediction).split())
            
                
        #print samples
        index = random.sample([i for i in range(len(hypotheses))], num_examples)
        for idx in index:
            print("Here is the model prediction")
            print(hypotheses[idx])
            print("-"*100)
            print("Here are what the references look like")
            for i in range(5):
                print(references[idx][i])
            print("-"*100)

        bleu_1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
        bleu_2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
        bleu_3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
        bleu_4 = corpus_bleu(references, hypotheses)
        meteor = corpus_meteor(references, hypotheses)
        
        writer.add_scalar('val_bleu1', bleu_1, epoch)
        writer.add_scalar('val_bleu2', bleu_2, epoch)
        writer.add_scalar('val_bleu3', bleu_3, epoch)
        writer.add_scalar('val_bleu4', bleu_4, epoch)
        writer.add_scalar('val_loss', np.mean(epoch_loss), epoch)
        writer.add_scalar('val_meteor', meteor, epoch)
        
        print(f'''Validation Epoch: {epoch}
              Val Loss: {np.mean(epoch_loss)}
              BLEU-1: {bleu_1}
              BLEU-2: {bleu_2}
              BLEU-3: {bleu_3}
              BLEU-4: {bleu_4}
              Meteor: {meteor}''')
    
    return bleu_4, global_step, np.mean(epoch_loss)
