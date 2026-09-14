"""
Gradio demo for the ViT + GPT-style image captioning model.

Runs on a free Hugging Face Gradio Space (CPU). It reuses the exact model
definition and inference routine from the training/Flask code so the demo
behaves identically to `app.py`.

Model weights (`best_meteor_model.pt`) are loaded from, in order:
  1. a local file if present (CHECKPOINT_PATH, useful for local development), else
  2. the public Hugging Face Hub repo set in MODEL_REPO_ID (the default), so the
     Space works without shipping the ~1.5 GB checkpoint in the app repo.
"""

import os

import torch
import gradio as gr
from transformers import AutoTokenizer, ViTImageProcessor

from vision.model.caption import VisionEncoderDecoder
from vision.inference.run import infer_caption


DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "best_meteor_model.pt")

MODEL_REPO_ID = os.environ.get(
    "MODEL_REPO_ID",
    "ngkuissi/vit-vit-large-patch16-224-in21k-gpt-2-w-cross-attention",
)
MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "best_meteor_model.pt")

HIDDEN_SIZE = 512
NUM_LAYERS = 12
NUM_HEADS = 8
MAX_LENGTH = 70


def _resolve_checkpoint() -> str:
    """Return a local path to the checkpoint, downloading from the Hub if configured."""
    if os.path.exists(CHECKPOINT_PATH):
        return CHECKPOINT_PATH
    if MODEL_REPO_ID:
        from huggingface_hub import hf_hub_download
        print(f"Downloading weights from Hub repo {MODEL_REPO_ID}/{MODEL_FILENAME}...")
        return hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
    raise FileNotFoundError(
        f"Checkpoint '{CHECKPOINT_PATH}' not found and MODEL_REPO_ID is not set. "
        "Upload best_meteor_model.pt to the Space (Git LFS) or set MODEL_REPO_ID."
    )


print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

print("Initializing model...")
model = VisionEncoderDecoder(
    vocab_size=tokenizer.vocab_size,
    max_length=MAX_LENGTH,
    num_layers=NUM_LAYERS,
    hidden_size=HIDDEN_SIZE,
    num_heads=NUM_HEADS,
)
model.to(DEVICE)

checkpoint_path = _resolve_checkpoint()
print(f"Loading weights from {checkpoint_path}...")
state_dict = torch.load(checkpoint_path, map_location=DEVICE)
# Support both raw state_dict and {'model_state_dict': ...} checkpoint formats.
if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
    state_dict = state_dict["model_state_dict"]
model.load_state_dict(state_dict)
model.eval()
print("Model loaded successfully.")

PROCESSOR = ViTImageProcessor.from_pretrained("google/vit-large-patch16-224-in21k")


# ----------------------------------------------------------------------------
# Inference function
# ----------------------------------------------------------------------------
def caption_image(image, temperature: float = 0.5) -> str:
    """Generate a caption for a PIL image using the loaded model."""
    if image is None:
        return "Please upload an image."

    image = image.convert("RGB")
    input_tensor = PROCESSOR(image, return_tensors="pt")["pixel_values"].squeeze(0)

    caption = infer_caption(
        model,
        input_tensor,
        tokenizer,
        DEVICE,
        max_length=MAX_LENGTH,
        temp=float(temperature),
    )
    return caption


# ----------------------------------------------------------------------------
# Gradio UI
# ----------------------------------------------------------------------------
DESCRIPTION = """
# 🖼️ Image Captioning with Vision Transformers

Upload an image and this model will describe it in natural language.

A **frozen ViT-Large encoder** extracts visual features and a **from-scratch,
GPT-2–style Transformer decoder** generates the caption one token at a time.
Trained end-to-end on **MS-COCO 2014**.

*Lower temperature → more literal, repeatable captions. Higher temperature → more varied wording.*
"""

demo = gr.Interface(
    fn=caption_image,
    inputs=[
        gr.Image(type="pil", label="Image"),
        gr.Slider(
            minimum=0.1,
            maximum=1.5,
            value=0.5,
            step=0.05,
            label="Temperature",
        ),
    ],
    outputs=gr.Textbox(label="Generated caption", lines=2),
    title="Vision Transformer Image Captioning",
    description=DESCRIPTION,
    allow_flagging="never",
)

if __name__ == "__main__":
    # server_name="0.0.0.0" is required so the app is reachable inside a Space.
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
