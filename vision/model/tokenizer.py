import os
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

TOKENIZER_NAME = "distilbert-base-uncased"
TOKENIZER = AutoTokenizer.from_pretrained("distilbert-base-uncased")