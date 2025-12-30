import torch


def load_model(model, checkpoint_path, device):
    '''
    Load model weights from a checkpoint
    Args:
        model (torch.nn.Module): The model to load weights into
        checkpoint_path (str): Path to the checkpoint file
        device (torch.device): Device to map the model weights
    '''
    print(f"Loading model weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    print("Model weights loaded successfully.")


def infer_caption(caption_model, image_tensor, tokenizer, device, max_length=90, temp=0.0):
    '''
    Generate caption for a given image tensor using the caption model.
    Args:
        caption_model (torch.nn.Module): The captioning model
        image_tensor (torch.Tensor): Preprocessed image tensor of shape [C, H, W]
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for decoding the output tokens 
        device (torch.device): Device to run the model on
        max_length (int): Maximum length of the generated caption
        temp (float): Sampling temperature for generation
    Returns:
        str: Generated caption
    '''
    caption_model.eval()
    
    image_tensor = image_tensor.to(device).unsqueeze(0)  # Add batch dimension
    with torch.inference_mode():
        generated_ids = caption_model.generate(
            input_image=image_tensor,
            sos_token=tokenizer.cls_token_id,
            eos_token=tokenizer.sep_token_id,
            pad_token=tokenizer.pad_token_id,
            max_length=max_length,
            temp=temp
        )
    caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return caption



    