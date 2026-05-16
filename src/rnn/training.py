import time
from contextlib import nullcontext
from math import ceil

from .experiment import (
    VARIATIONS,
    load_caption_sequences,
    load_feature_map,
    load_or_create_feature_scaler,
    load_text_util,
    scale_feature_map,
    split_image_keys,
)
from .modeling import build_caption_model
from .paths import ARCH_TAG, RnnPaths
from .utils.train_utils import DataGenerator, make_caption_dataset


def prepare_training_data(repo_root=None, train_size=6000, val_size=1000, seed=42, sequence_length=35):
    paths = RnnPaths.from_root(repo_root)
    text_util = load_text_util(paths.root, sequence_length=sequence_length)
    _, sequence_mapping = load_caption_sequences(paths.root, text_util)
    raw_image_features = load_feature_map(paths.root)

    image_keys = [key for key in sequence_mapping if key in raw_image_features]
    train_keys, val_keys, test_keys = split_image_keys(
        image_keys,
        train_size=train_size,
        val_size=val_size,
        seed=seed,
    )
    mean, std = load_or_create_feature_scaler(paths.root, raw_image_features, train_keys)
    image_features = scale_feature_map(raw_image_features, mean, std)

    return {
        "text_util": text_util,
        "image_features": image_features,
        "feature_mean": mean,
        "feature_std": std,
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
    strategy=None,
    use_tf_dataset=False,
    retrain_existing=False,
):
    pd = _pd()
    paths = RnnPaths.from_root(repo_root)
    paths.weights_dir.mkdir(parents=True, exist_ok=True)
    paths.results_dir.mkdir(parents=True, exist_ok=True)

    variations = variations or VARIATIONS
    previous_history = _load_previous_history(pd, paths.training_history_file)

    if (
        not retrain_existing
        and not previous_history.empty
        and _history_covers_expected(previous_history, variations)
        and _all_expected_weights_exist(paths, variations)
    ):
        return _ordered_history(pd, previous_history, variations, paths)

    data = None
    text_util = None
    previous_rows = _history_rows_by_key(previous_history)
    rows = []

    for model_type in ("SimpleRNN", "LSTM"):
        is_lstm = model_type == "LSTM"
        for variation in variations:
            weight_path = paths.weight_file(model_type, variation["layers"], variation["hidden_state"])
            key = _history_key(model_type, variation)
            previous_row = previous_rows.get(key)

            if weight_path.exists() and not retrain_existing:
                rows.append(_skipped_row(previous_row, model_type, variation, weight_path))
                print(
                    f"[skip] {model_type} {variation['variation_name']} sudah punya weight: {weight_path.name}",
                    flush=True,
                )
                _save_training_history(pd, paths.training_history_file, rows)
                continue

            if data is None:
                data = prepare_training_data(paths.root, sequence_length=sequence_length)
                text_util = data["text_util"]

            print(
                f"[train] {model_type} {variation['variation_name']} -> {weight_path.name}",
                flush=True,
            )

            scope = strategy.scope() if strategy is not None else nullcontext()
            with scope:
                model = build_caption_model(
                    is_lstm=is_lstm,
                    layers=variation["layers"],
                    hidden_state=variation["hidden_state"],
                    vocab_size=text_util.vocab_size,
                    sequence_length=text_util.sequence_length,
                )

            train_gen = _training_source(
                mapping_data=data["train_data"],
                image_features=data["image_features"],
                sequence_length=text_util.sequence_length,
                batch_size=batch_size,
                pad_id=text_util.word_to_idx.get("pad", 0),
                use_tf_dataset=use_tf_dataset,
            )
            val_gen = _training_source(
                mapping_data=data["val_data"],
                image_features=data["image_features"],
                sequence_length=text_util.sequence_length,
                batch_size=batch_size,
                pad_id=text_util.word_to_idx.get("pad", 0),
                use_tf_dataset=use_tf_dataset,
            )

            start = time.perf_counter()
            history = model.fit(
                train_gen,
                steps_per_epoch=max(ceil(len(data["train_data"]) / batch_size), 1),
                validation_data=val_gen,
                validation_steps=max(ceil(len(data["val_data"]) / batch_size), 1),
                epochs=epochs,
                verbose=1,
            )
            elapsed = time.perf_counter() - start

            model.save_weights(str(weight_path))
            rows.append(
                {
                    "architecture": ARCH_TAG,
                    "model_type": model_type,
                    "variation_name": variation["variation_name"],
                    "layers": variation["layers"],
                    "hidden_state": variation["hidden_state"],
                    "final_loss": history.history["loss"][-1],
                    "final_val_loss": history.history["val_loss"][-1],
                    "history_loss": history.history["loss"],
                    "history_val_loss": history.history["val_loss"],
                    "training_time_sec": elapsed,
                    "weight_file": weight_path.name,
                    "training_status": "trained",
                }
            )
            _save_training_history(pd, paths.training_history_file, rows)

    return _save_training_history(pd, paths.training_history_file, rows)


def training_artifacts_complete(repo_root=None, variations=None):
    pd = _pd()
    paths = RnnPaths.from_root(repo_root)
    variations = variations or VARIATIONS
    if not paths.training_history_file.exists():
        return False

    history = _load_previous_history(pd, paths.training_history_file)
    return _history_covers_expected(history, variations) and _all_expected_weights_exist(paths, variations)


def _select_mapping(mapping, keys):
    return {key: mapping[key] for key in keys}


def _training_source(mapping_data, image_features, sequence_length, batch_size, pad_id, use_tf_dataset):
    if use_tf_dataset:
        return make_caption_dataset(
            mapping_data=mapping_data,
            image_features=image_features,
            sequence_length=sequence_length,
            batch_size=batch_size,
            pad_id=pad_id,
        )

    return DataGenerator(
        mapping_data=mapping_data,
        image_features=image_features,
        sequence_length=sequence_length,
        batch_size=batch_size,
        pad_id=pad_id,
    ).generate()


def _pd():
    import pandas as pd

    return pd


def _assert_context_v2(frame, path):
    if "architecture" not in frame.columns:
        raise ValueError(f"{path} adalah artifact lama tanpa kolom architecture; regenerate {ARCH_TAG}.")
    invalid = set(frame["architecture"].dropna().astype(str)) - {ARCH_TAG}
    if invalid:
        raise ValueError(f"{path} bukan artifact {ARCH_TAG}: {sorted(invalid)}")


def _load_previous_history(pd, path):
    if not path.exists():
        return pd.DataFrame()
    history = pd.read_csv(path)
    _assert_context_v2(history, path)
    return history


def _history_key(model_type, variation):
    return (
        str(model_type),
        str(variation["variation_name"]),
        int(variation["layers"]),
        int(variation["hidden_state"]),
    )


def _history_key_from_row(row):
    return (
        str(row["model_type"]),
        str(row["variation_name"]),
        int(row["layers"]),
        int(row["hidden_state"]),
    )


def _history_rows_by_key(history):
    if history.empty:
        return {}
    rows = {}
    for row in history.to_dict("records"):
        rows[_history_key_from_row(row)] = row
    return rows


def _expected_keys(variations):
    return {
        _history_key(model_type, variation)
        for model_type in ("SimpleRNN", "LSTM")
        for variation in variations
    }


def _history_covers_expected(history, variations):
    if history.empty:
        return False
    return _expected_keys(variations).issubset(set(_history_rows_by_key(history)))


def _all_expected_weights_exist(paths, variations):
    return all(
        paths.weight_file(model_type, variation["layers"], variation["hidden_state"]).exists()
        for model_type in ("SimpleRNN", "LSTM")
        for variation in variations
    )


def _ordered_history(pd, history, variations, paths=None):
    rows_by_key = _history_rows_by_key(history)
    rows = []
    for model_type in ("SimpleRNN", "LSTM"):
        for variation in variations:
            row = dict(rows_by_key[_history_key(model_type, variation)])
            if paths is not None:
                weight_path = paths.weight_file(model_type, variation["layers"], variation["hidden_state"])
                if _is_blank(row.get("weight_file")):
                    row["weight_file"] = weight_path.name
                if _is_blank(row.get("training_status")):
                    row["training_status"] = "loaded_history"
            rows.append(row)
    return pd.DataFrame(rows)


def _skipped_row(previous_row, model_type, variation, weight_path):
    if previous_row is not None:
        row = dict(previous_row)
        if _is_blank(row.get("weight_file")):
            row["weight_file"] = weight_path.name
        if _is_blank(row.get("training_status")):
            row["training_status"] = "skipped_existing_weight"
        return row

    return {
        "architecture": ARCH_TAG,
        "model_type": model_type,
        "variation_name": variation["variation_name"],
        "layers": variation["layers"],
        "hidden_state": variation["hidden_state"],
        "final_loss": None,
        "final_val_loss": None,
        "history_loss": None,
        "history_val_loss": None,
        "training_time_sec": None,
        "weight_file": weight_path.name,
        "training_status": "skipped_existing_weight",
    }


def _save_training_history(pd, path, rows):
    history_df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(path, index=False)
    return history_df


def _is_blank(value):
    return value is None or value != value or value == ""
