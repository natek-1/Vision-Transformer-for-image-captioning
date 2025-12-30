from flask import Flask, render_template, request, jsonify
import os
import torch
from PIL import Image
import base64
import io
import torchvision.transforms as transforms
from transformers import AutoTokenizer

from vision.model.caption import VisionEncoderDecoder
from vision.inference.run import infer_caption

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Model & Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoint.pt" 

# Hyperparameters (must match training)
IMAGE_SIZE = 128
HIDDEN_SIZE = 192
NUM_LAYERS = (6, 6)
NUM_HEADS = 8
PATCH_SIZE = 8
MAX_LENGTH = 90

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

print("Initializing model...")
model = VisionEncoderDecoder(image_size=IMAGE_SIZE, channels_in=3, num_emb=tokenizer.vocab_size,
                             patch_size=PATCH_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
                             num_heads=NUM_HEADS, mlp_dropout=0.1, att_dropout=0.1)

model.to(DEVICE)

if os.path.exists(CHECKPOINT_PATH):
    print(f"Loading weights from {CHECKPOINT_PATH}...")
    try:
        state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state_dict)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        # Fallback
        try:
           checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
           if 'model_state_dict' in checkpoint:
               model.load_state_dict(checkpoint['model_state_dict'])
               print("Model loaded from checkpoint dict successfully.")
        except Exception as e:
           print("Fatal error loading model.", e)
else:
    print(f"Warning: {CHECKPOINT_PATH} not found. Running with random weights.")

model.eval()

# --- Preprocessing ---
val_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_caption', methods=['POST'])
def generate_caption():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Read generation parameters
    # User requested temperature close to 0.5. Defaulting to 0.5.
    try:
        temperature = float(request.form.get('temperature', 0.5))
    except ValueError:
        temperature = 0.5
    
    try:
        # Process the image
        image = Image.open(file.stream).convert('RGB')
        
        # Preprocess
        input_tensor = val_transform(image)
        
        # Generate caption
        caption = infer_caption(model, input_tensor, tokenizer, DEVICE, max_length=MAX_LENGTH, temp=temperature)
        
        # Convert the original image to base64 to display
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return jsonify({
            'caption': caption,
            'image': img_str,
            'parameters': {
                'temperature': temperature,
            }
        })
        
    except Exception as e:
        print(f"Error generating caption: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)