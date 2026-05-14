import time

import numpy as np

from .experiment import VARIATIONS, load_caption_sequences, load_feature_map, load_text_util
from .modeling import build_caption_model
from .paths import RnnPaths
from .utils.train_utils import DataGenerator


def prepare_training_data(repo_root=None, train_size=6000, val_size=1000, seed=42, sequence_length=35):
    paths = RnnPaths.from_root(repo_root)
    text_util = load_text_util(paths.root, sequence_length=sequence_length)
    _, sequence_mapping = load_caption_sequences(paths.root, text_util)
    image_features = load_feature_map(paths.root)

    image_keys = [key for key in sequence_mapping if key in image_features]
    rng = np.random.default_rng(seed)
    rng.shuffle(image_keys)

    train_keys = image_keys[:train_size]
    val_keys = image_keys[train_size:train_size + val_size]
    test_keys = image_keys[train_size + val_size:]

    return {
        "text_util": text_util,
        "image_features": image_features,
        "train_data": _select_mapping(sequence_mapping, train_keys),
        "val_data": _select_mapping(sequence_mapping, val_keys),
        "test_data": _select_mapping(sequence_mapping, test_keys),
        "train_keys": train_keys,
        "val_keys": val_keys,
        "test_keys": test_keys,
    }


def train_all_variations(
    repo_root=None,
    epochs=5,
    batch_size=64,
    sequence_length=35,
    variations=None,
    force=False,
):
    pd = _pd()
    paths = RnnPaths.from_root(repo_root)
    paths.weights_dir.mkdir(parents=True, exist_ok=True)
    paths.results_dir.mkdir(parents=True, exist_ok=True)

    if paths.training_history_file.exists() and not force:
        return pd.read_csv(paths.training_history_file)

    data = prepare_training_data(paths.root, sequence_length=sequence_length)
    text_util = data["text_util"]
    variations = variations or VARIATIONS
    rows = []

    for model_type in ("SimpleRNN", "LSTM"):
        is_lstm = model_type == "LSTM"
        for variation in variations:
            model = build_caption_model(
                is_lstm=is_lstm,
                layers=variation["layers"],
                hidden_state=variation["hidden_state"],
                vocab_size=text_util.vocab_size,
                sequence_length=text_util.sequence_length,
            )

            train_gen = DataGenerator(
                mapping_data=data["train_data"],
                image_features=data["image_features"],
                vocab_size=text_util.vocab_size,
                sequence_length=text_util.sequence_length,
                batch_size=batch_size,
            ).generate()
            val_gen = DataGenerator(
                mapping_data=data["val_data"],
                image_features=data["image_features"],
                vocab_size=text_util.vocab_size,
                sequence_length=text_util.sequence_length,
                batch_size=batch_size,
            ).generate()

            start = time.perf_counter()
            history = model.fit(
                train_gen,
                steps_per_epoch=max(len(data["train_data"]) // batch_size, 1),
                validation_data=val_gen,
                validation_steps=max(len(data["val_data"]) // batch_size, 1),
                epochs=epochs,
                verbose=1,
            )
            elapsed = time.perf_counter() - start

            model.save_weights(str(paths.weight_file(model_type, variation["layers"], variation["hidden_state"])))
            rows.append(
                {
                    "model_type": model_type,
                    "variation_name": variation["variation_name"],
                    "layers": variation["layers"],
                    "hidden_state": variation["hidden_state"],
                    "final_loss": history.history["loss"][-1],
                    "final_val_loss": history.history["val_loss"][-1],
                    "history_loss": history.history["loss"],
                    "history_val_loss": history.history["val_loss"],
                    "training_time_sec": elapsed,
                }
            )

    history_df = pd.DataFrame(rows)
    history_df.to_csv(paths.training_history_file, index=False)
    return history_df


def _select_mapping(mapping, keys):
    return {key: mapping[key] for key in keys}


def _pd():
    import pandas as pd

    return pd
