from pathlib import Path

import numpy as np
from PIL import Image
from .paths import RnnPaths


def build_inception_encoder():
    from tensorflow.keras.applications import InceptionV3
    from tensorflow.keras.models import Model

    base_model = InceptionV3(weights="imagenet")
    base_model.trainable = False
    return Model(inputs=base_model.input, outputs=base_model.get_layer("avg_pool").output)


def load_inception_image(image_path, target_size=(299, 299)):
    from tensorflow.keras.applications.inception_v3 import preprocess_input

    img = Image.open(image_path).convert("RGB").resize(target_size)
    arr = np.asarray(img, dtype=np.float32)
    return preprocess_input(np.expand_dims(arr, axis=0))


def extract_features(image_paths, encoder=None, batch_size=32):
    if encoder is None:
        encoder = build_inception_encoder()

    features = []
    names = []
    batch = []
    batch_names = []

    for image_path in image_paths:
        image_path = Path(image_path)
        batch.append(load_inception_image(image_path)[0])
        batch_names.append(image_path.name)

        if len(batch) == batch_size:
            pred = encoder.predict(np.asarray(batch), verbose=0)
            features.append(pred.reshape(pred.shape[0], -1))
            names.extend(batch_names)
            batch = []
            batch_names = []

    if batch:
        pred = encoder.predict(np.asarray(batch), verbose=0)
        features.append(pred.reshape(pred.shape[0], -1))
        names.extend(batch_names)

    return np.vstack(features), np.asarray(names)


def extract_and_save_features(image_dir, output_dir, batch_size=32, force=False):
    output_dir = Path(output_dir)
    features_path = output_dir / "features.npy"
    names_path = output_dir / "image_names.npy"
    if not force and features_path.exists() and names_path.exists():
        return np.load(features_path), np.load(names_path, allow_pickle=True)

    image_paths = sorted(Path(image_dir).glob("*.jpg"))
    output_dir.mkdir(parents=True, exist_ok=True)
    features, names = extract_features(image_paths, batch_size=batch_size)
    np.save(features_path, features)
    np.save(names_path, names)
    return features, names


def extract_and_save_repo_features(repo_root=None, batch_size=32, force=False):
    paths = RnnPaths.from_root(repo_root)
    return extract_and_save_features(paths.image_dir, paths.feature_dir, batch_size=batch_size, force=force)
