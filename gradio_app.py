import os

import torch
import gradio as gr
from transformers import AutoTokenizer, ViTImageProcessor

from vision.model.caption import VisionEncoderDecoder
from vision.inference.run import infer_caption

try:
    import spaces  # provided by the ZeroGPU runtime

    _ZEROGPU = True
except Exception:  # not on ZeroGPU (local dev, CPU Space, etc.)
    _ZEROGPU = False

    class _SpacesShim:
        @staticmethod
        def GPU(*args, **kwargs):
            # Support both @spaces.GPU and @spaces.GPU(duration=...) forms.
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]

            def _decorator(fn):
                return fn

            return _decorator

    spaces = _SpacesShim()


def _patch_gradio_client_bool_schema() -> None:
    try:
        import gradio_client.utils as _gcu
    except Exception:
        return

    _orig_get_type = getattr(_gcu, "get_type", None)
    if _orig_get_type is not None:
        def _safe_get_type(schema):
            if not isinstance(schema, dict):
                return "Any"
            return _orig_get_type(schema)
        _gcu.get_type = _safe_get_type

    _orig_j2p = getattr(_gcu, "_json_schema_to_python_type", None)
    if _orig_j2p is not None:
        def _safe_j2p(schema, defs=None):
            if not isinstance(schema, dict):
                return "Any"
            return _orig_j2p(schema, defs)
        _gcu._json_schema_to_python_type = _safe_j2p


_patch_gradio_client_bool_schema()

print(f"gradio version in use: {gr.__version__}")

if _ZEROGPU:
    DEVICE = "cuda"
else:
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


@spaces.GPU(duration=60)
def caption_image(image, temperature: float = 0.5) -> str:
    """Generate a caption for a PIL image using the loaded model.

    Decorated with @spaces.GPU so ZeroGPU allocates a GPU for the duration of
    the call. The decorator is a no-op (via the shim above) off ZeroGPU.
    """
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


DESCRIPTION = """
# 🖼️ Image Captioning with Vision Transformers

Upload an image and this model will describe it in natural language.
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
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        show_api=False,
    )
