import numpy as np
from PIL import Image
from typing import List


def load_image(file_path: str, target_size: tuple = (150, 150)) -> np.ndarray:
    img = Image.open(file_path).convert('RGB')
    img = img.resize((target_size[1], target_size[0]))
    arr = np.array(img, dtype=np.float32) / 255.0
    
    return arr


def load_batch(file_paths: List[str], target_size: tuple = (150, 150)) -> np.ndarray:
    images = [load_image(p, target_size) for p in file_paths]
    return np.stack(images, axis=0)
