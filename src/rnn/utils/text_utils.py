import re
import os
import json
import numpy as np

class CaptionPreprocessor:
    def __init__(self, sequence_length=35):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        self.sequence_length = sequence_length

    def load_captions(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} tidak ditemukan.")
        with open(file_path, "r", encoding="utf-8") as f:
            return [caption.lower().strip() for caption in f.readlines()[1:]]
    
    def clean_text(self, text):
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def build_vocabulary(self, file_path, force_build=False, directory="output"):
        word_to_idx_path = os.path.join(directory, "vocab_word_to_idx.json")
        idx_to_word_path = os.path.join(directory, "vocab_idx_to_word.json")

        if not force_build and os.path.exists(word_to_idx_path) and os.path.exists(idx_to_word_path):
            with open(word_to_idx_path, "r", encoding="utf-8") as f:
                self.word_to_idx = json.load(f)
            with open(idx_to_word_path, "r", encoding="utf-8") as f:
                self.idx_to_word = {int(k): v for k, v in json.load(f).items()}
            self._ensure_special_tokens()
            return None
        
        captions_raw = self.load_captions(file_path)
        cleaned = [
            f"<start> {self.clean_text(caption.split(',', 1)[1])} <end>"
            for caption in captions_raw
            if "," in caption
        ]

        words = []
        for caption in cleaned:
            words.extend(caption.replace("<", "").replace(">", "").split())

        unique_words = ["unk"] + sorted(set(words) - {"pad", "unk"})
        self.word_to_idx = {"pad": 0}
        self.word_to_idx.update({word: idx + 1 for idx, word in enumerate(unique_words)})
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self._ensure_special_tokens()
        
        self.save(directory) 
        return cleaned
    
    def get_image_to_captions_mapping(self, file_path):
        captions_raw = self.load_captions(file_path)
        mapping = {}

        for line in captions_raw:
            parts = line.split(",", 1)
            if len(parts) < 2:
                continue
            
            image_name = parts[0].strip()
            caption_text = f"<start> {self.clean_text(parts[1])} <end>"
            
            if image_name not in mapping:
                mapping[image_name] = []
            mapping[image_name].append(caption_text)
            
        return mapping

    def texts_to_sequences(self, captions):
        sequences = []
        for caption in captions:
            words = caption.replace("<", "").replace(">", "").split()
            seq = [self.word_to_idx.get(word, self.word_to_idx.get("unk")) for word in words]
            sequences.append(seq)
        return sequences

    def pad_sequences(self, sequences):
        padded_sequences = []
        for seq in sequences:
            if len(seq) < self.sequence_length:
                new_seq = seq + [self.word_to_idx.get("pad", 0)] * (self.sequence_length - len(seq))
            else:
                new_seq = seq[:self.sequence_length]
            padded_sequences.append(new_seq)
        return np.array(padded_sequences)
    
    def save(self, directory="output"):
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        word_to_idx_path = os.path.join(directory, "vocab_word_to_idx.json")
        idx_to_word_path = os.path.join(directory, "vocab_idx_to_word.json")

        with open(word_to_idx_path, "w", encoding="utf-8") as f:
            json.dump(self.word_to_idx, f, indent=4)
        
        with open(idx_to_word_path, "w", encoding="utf-8") as f:
            json.dump(self.idx_to_word, f, indent=4)
            
        print(f"Vocabulary berhasil disimpan di folder: {directory}")

    def _ensure_special_tokens(self):
        if "pad" not in self.word_to_idx:
            self.word_to_idx["pad"] = 0
        if 0 not in self.idx_to_word:
            self.idx_to_word[0] = "pad"

        next_idx = max([int(v) for v in self.word_to_idx.values()], default=0) + 1
        for token in ("unk", "start", "end"):
            if token not in self.word_to_idx:
                self.word_to_idx[token] = next_idx
                self.idx_to_word[next_idx] = token
                next_idx += 1

        self.vocab_size = max(int(v) for v in self.word_to_idx.values()) + 1
