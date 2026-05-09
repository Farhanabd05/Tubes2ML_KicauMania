import numpy as np
from typing import List
from pathlib import Path
from .image_utils import load_batch

class FeatureExtractor:

    def __init__(self, keras_encoder, target_size: tuple = (150, 150)):
        self.keras_encoder = keras_encoder
        self.target_size = target_size

    def extract_and_save(self, file_paths: List[str], save_path: str,
                         batch_size: int = 32) -> np.ndarray:
        save_path_obj = Path(save_path)
        
        if save_path_obj.exists():
            print(f"File fitur sudah ada di {save_path}, memuat dari disk...")
            return self.load_features(save_path)
            
        print(f"Mengekstrak fitur untuk {len(file_paths)} gambar...")
        features_list = []
        
        # Proses per batch
        for i in range(0, len(file_paths), batch_size):
            batch_paths = file_paths[i:i + batch_size]
            batch_images = load_batch(batch_paths, self.target_size)
            
            batch_features = self.keras_encoder.predict(batch_images, verbose=0)
            
            batch_features = batch_features.reshape(batch_images.shape[0], -1)
            
            features_list.append(batch_features)
            
            if (i // batch_size) % 10 == 0:
                print(f"Telah memproses {i}/{len(file_paths)} gambar...")
                
        all_features = np.vstack(features_list)
        
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(save_path, all_features)
        print(f"Berhasil mengekstrak dan menyimpan fitur ke {save_path}")
        
        return all_features

    def load_features(self, save_path: str) -> np.ndarray:
        return np.load(save_path)
