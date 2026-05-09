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
    
    def build_vocabulary(self, file_path):
        captions = self.load_captions(file_path)
        cleaned = [f"<start> {self.clean_text(caption.split(',')[1])} <end>" for caption in captions]

        tokenizer = Tokenizer(oov_token="<unk>")
        tokenizer.fit_on_texts(cleaned)
        
        self.word_to_idx = tokenizer.word_index
        self.idx_to_word = tokenizer.index_word

        self.vocab_size = len(tokenizer.word_index) + 1
        return cleaned

    def save(self, directory="output"):
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        path = os.path.join(directory, "vocab.json")
        with open(path, 'w') as f:
            json.dump(self.word_to_idx, f)
        print(f"Vocabulary berhasil disimpan ke {path}")

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