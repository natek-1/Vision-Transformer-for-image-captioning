
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class Vocabulary():
    max_len = 45

    
    def __init__(self, model_name = "gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    
    def __len__(self):
        return self.tokenizer.vocab_size

    def vocab_size(self):
        return self.tokenizer.vocab_size
    
    def encode(self, text, with_attention_mask: bool = False, padding = False):
        max_seq = None
        if padding:
            max_seq = Vocabulary.max_len
            padding = 'max_length'
        return self.tokenizer(text, padding=padding, max_length=max_seq, return_attention_mask=with_attention_mask)
    
    def decode(self, text, skip_special_tokens=True):
        return self.tokenizer.decode(text, skip_special_tokens=skip_special_tokens)