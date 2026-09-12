from torch._C import _VariableFunctions
from flask import Flask, render_template, request, jsonify
import os
import torch
from PIL import Image
import base64
import io
import torchvision.transforms as transforms
from transformers import AutoTokenizer
from transformers import ViTImageProcessor

from vision.model.caption import VisionEncoderDecoder
from vision.inference.run import infer_caption

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
CHECKPOINT_PATH = "best_meteor_model.pt" 

HIDDEN_SIZE = 512
NUM_LAYERS = 12
NUM_HEADS = 8
MAX_LENGTH = 70

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

print("Initializing model...")
model = VisionEncoderDecoder(vocab_size=tokenizer.vocab_size, max_length=MAX_LENGTH, 
                            num_layers=NUM_LAYERS, hidden_size=HIDDEN_SIZE, 
                            num_heads=NUM_HEADS)

model.to(DEVICE)

if os.path.exists(CHECKPOINT_PATH):
    print(f"Loading weights from {CHECKPOINT_PATH}...")
    try:
        state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("Model loaded from checkpoint dict successfully.")
        except Exception as e:
            raise ValueError(f"Fatal error loading model. Error: {e}")
else:
    raise FileNotFoundError(f"Error: {CHECKPOINT_PATH} not found.")

model.eval()

PROCESSOR = ViTImageProcessor.from_pretrained('google/vit-large-patch16-224-in21k')

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
        input_tensor = PROCESSOR(image, return_tensors="pt")["pixel_values"].squeeze(0)
        
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