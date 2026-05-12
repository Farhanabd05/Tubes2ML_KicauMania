import re
import os
import json
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

class CaptionPreprocessor:
    def __init__(self, sequence_length=35):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        self.sequence_length = sequence_length

    def load_captions(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} tidak ditemukan.")
        with open(file_path, 'r') as f:
            captions = f.readlines()
            captions = [caption.lower() for caption in captions[1:]]
        return captions
    
    def clean_text(self, text):
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def build_vocabulary(self, file_path, force_build=False, directory="output"):
        word_to_idx_path = os.path.join(directory, "vocab_word_to_idx.json")
        idx_to_word_path = os.path.join(directory, "vocab_idx_to_word.json")

        if not force_build and os.path.exists(word_to_idx_path) and os.path.exists(idx_to_word_path):
            with open(word_to_idx_path, 'r') as f:
                self.word_to_idx = json.load(f)
            with open(idx_to_word_path, 'r') as f:
                self.idx_to_word = {int(k): v for k, v in json.load(f).items()} # JSON key selalu string
            self.vocab_size = len(self.word_to_idx) + 1
            return None
        
        captions_raw = self.load_captions(file_path)
        cleaned = [f"<start> {self.clean_text(c.split(',')[1])} <end>" for c in captions_raw]

        tokenizer = Tokenizer(oov_token="<unk>")
        tokenizer.fit_on_texts(cleaned)
        
        self.word_to_idx = tokenizer.word_index
        self.idx_to_word = tokenizer.index_word
        self.vocab_size = len(tokenizer.word_index) + 1
        
        self.save(directory) 
        return cleaned
    
    def get_image_to_captions_mapping(self, file_path):
        captions_raw = self.load_captions(file_path)
        mapping = {}

        for line in captions_raw:
            parts = line.split(',')
            if len(parts) < 2: continue
            
            image_name = parts[0].strip()
            caption_text = f"<start> {self.clean_text(parts[1])} <end>"
            
            if image_name not in mapping:
                mapping[image_name] = []
            mapping[image_name].append(caption_text)
            
        return mapping

    def texts_to_sequences(self, captions):
        sequences = []
        for caption in captions:
            words = caption.split()
            seq = [self.word_to_idx.get(word, self.word_to_idx.get("<unk>")) for word in words]
            sequences.append(seq)
        return sequences

    def pad_sequences(self, sequences):
        padded_sequences = []
        for seq in sequences:
            if len(seq) < self.sequence_length:
                new_seq = seq + [0] * (self.sequence_length - len(seq))
            else:
                new_seq = seq[:self.sequence_length]
            padded_sequences.append(new_seq)
        return np.array(padded_sequences)
    
    def save(self, directory="output"):
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        word_to_idx_path = os.path.join(directory, "vocab_word_to_idx.json")
        idx_to_word_path = os.path.join(directory, "vocab_idx_to_word.json")

        with open(word_to_idx_path, 'w') as f:
            json.dump(self.word_to_idx, f, indent=4)
        
        # Simpan idx_to_word
        with open(idx_to_word_path, 'w') as f:
            json.dump(self.idx_to_word, f, indent=4)
            
        print(f"Vocabulary berhasil disimpan di folder: {directory}")