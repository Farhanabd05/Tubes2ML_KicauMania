import ast
import json
import time
from pathlib import Path

import numpy as np

from .ImageCaptioningScratch import ImageCaptioningModel
from .modeling import build_caption_model, greedy_decode_keras
from .utils.text_utils import CaptionPreprocessor


VARIATIONS = [
    {"variation_name": "Shallow_Small", "layers": 1, "hidden_state": 128},
    {"variation_name": "Deep_Small", "layers": 2, "hidden_state": 128},
    {"variation_name": "VeryDeep_Small", "layers": 3, "hidden_state": 128},
    {"variation_name": "Shallow_Mid", "layers": 1, "hidden_state": 256},
    {"variation_name": "Shallow_Large", "layers": 1, "hidden_state": 512},
    {"variation_name": "Deep_Large", "layers": 2, "hidden_state": 512},
]


def find_repo_root(start=None):
    cwd = Path(start or Path.cwd()).resolve()
    for path in [cwd, *cwd.parents]:
        if (path / "src" / "rnn").exists() and (path / "RNN_dataset").exists():
            return path
    return cwd


def load_feature_map(repo_root):
    feature_dir = Path(repo_root) / "images_feature"
    features = np.load(feature_dir / "features.npy")
    names = np.load(feature_dir / "image_names.npy", allow_pickle=True)
    return {str(name): features[idx] for idx, name in enumerate(names)}


def load_text_util(repo_root, sequence_length=35, force_build=False):
    text_util = CaptionPreprocessor(sequence_length=sequence_length)
    vocab_dir = Path(repo_root) / "src" / "rnn" / "output"
    text_util.build_vocabulary(
        str(Path(repo_root) / "RNN_dataset" / "captions.txt"),
        force_build=force_build,
        directory=str(vocab_dir),
    )
    return text_util


def load_caption_sequences(repo_root, text_util):
    mapping = text_util.get_image_to_captions_mapping(str(Path(repo_root) / "RNN_dataset" / "captions.txt"))
    sequence_mapping = {}
    for image_name, captions in mapping.items():
        sequence_mapping[image_name] = text_util.pad_sequences(text_util.texts_to_sequences(captions))
    return mapping, sequence_mapping


def split_image_keys(keys, train_size=6000, val_size=1000):
    keys = sorted(keys)
    return keys[:train_size], keys[train_size:train_size + val_size], keys[train_size + val_size:]


def weight_path(repo_root, model_type, variation_name, layers, hidden_state):
    prefix = "LSTM" if model_type == "LSTM" else "SimpleRNN"
    layer_tag = {1: "Shallow", 2: "Deep", 3: "VeryDeep"}.get(int(layers), f"L{layers}")
    size_tag = "Small" if int(hidden_state) == 128 else "Mid" if int(hidden_state) == 256 else "Large"
    filename = f"{prefix}_{layer_tag}_{size_tag}_L{int(layers)}_H{int(hidden_state)}.h5"
    return Path(repo_root) / "artifacts" / "rnn" / "weights" / filename


def make_keras_model(repo_root, text_util, model_type, variation_name, layers, hidden_state):
    model = build_caption_model(
        is_lstm=model_type == "LSTM",
        layers=int(layers),
        hidden_state=int(hidden_state),
        vocab_size=text_util.vocab_size,
        sequence_length=text_util.sequence_length,
    )
    path = weight_path(repo_root, model_type, variation_name, layers, hidden_state)
    model.load_weights(str(path))
    return model


def score_caption(references, hypothesis):
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    ref_tokens = [clean_generated_caption(ref).split() for ref in references]
    hyp_tokens = clean_generated_caption(hypothesis).split()
    smoothing = SmoothingFunction().method1
    bleu_4 = sentence_bleu(ref_tokens, hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)

    try:
        from nltk.translate.meteor_score import meteor_score

        meteor = meteor_score(ref_tokens, hyp_tokens)
    except Exception:
        meteor = np.nan

    return bleu_4, meteor


def clean_generated_caption(text):
    banned = {"pad", "start", "end", "<pad>", "<start>", "<end>", "unk", "<unk>"}
    return " ".join(word for word in str(text).split() if word not in banned)


def evaluate_decoder(model, image_keys, image_features, reference_mapping, decoder, max_len=35, limit=None):
    pd = _pd()
    rows = []
    selected_keys = list(image_keys[:limit] if limit else image_keys)
    start = time.perf_counter()

    for image_name in selected_keys:
        if image_name not in image_features:
            continue
        feature = image_features[image_name]
        generated = decoder(model, feature, max_len=max_len)
        bleu_4, meteor = score_caption(reference_mapping[image_name], generated)
        rows.append(
            {
                "image_name": image_name,
                "generated_caption": generated,
                "bleu_4": bleu_4,
                "meteor": meteor,
            }
        )

    elapsed = time.perf_counter() - start
    avg_time = elapsed / max(len(rows), 1)
    return pd.DataFrame(rows), elapsed, avg_time


def evaluate_all_variations(repo_root=None, split="test", limit=None, max_len=35):
    pd = _pd()
    repo_root = find_repo_root(repo_root)
    text_util = load_text_util(repo_root)
    reference_mapping, _ = load_caption_sequences(repo_root, text_util)
    image_features = load_feature_map(repo_root)
    _, val_keys, test_keys = split_image_keys([k for k in reference_mapping if k in image_features])
    keys = test_keys if split == "test" else val_keys

    history_path = Path(repo_root) / "src" / "rnn" / "hasil_variasi_model.csv"
    history = pd.read_csv(history_path)
    result_rows = []
    detail_frames = []

    for row in history.to_dict("records"):
        model = make_keras_model(
            repo_root,
            text_util,
            row["model_type"],
            row["variation_name"],
            row["layers"],
            row["hidden_state"],
        )
        scratch_model = ImageCaptioningModel(model, text_util, is_lstm=row["model_type"] == "LSTM")
        detail, elapsed, avg_time = evaluate_decoder(
            scratch_model,
            keys,
            image_features,
            reference_mapping,
            decoder=lambda m, feature, max_len: m.generate_caption(feature, max_len=max_len),
            max_len=max_len,
            limit=limit,
        )
        detail.insert(0, "model_type", row["model_type"])
        detail.insert(1, "variation_name", row["variation_name"])
        detail_frames.append(detail)

        result_rows.append(
            {
                "model_type": row["model_type"],
                "variation_name": row["variation_name"],
                "layers": int(row["layers"]),
                "hidden_state": int(row["hidden_state"]),
                "scratch_bleu_4": detail["bleu_4"].mean(),
                "scratch_meteor": detail["meteor"].mean(),
                "scratch_total_time_sec": elapsed,
                "scratch_avg_time_sec": avg_time,
                "training_time_sec": row.get("training_time_sec"),
                "final_loss": row.get("final_loss"),
                "final_val_loss": row.get("final_val_loss"),
            }
        )

    results = pd.DataFrame(result_rows)
    details = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    output_dir = Path(repo_root) / "artifacts" / "rnn" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "scratch_variation_eval.csv", index=False)
    details.to_csv(output_dir / "scratch_caption_details.csv", index=False)
    return results, details


def compare_best_keras_vs_scratch(repo_root=None, split="test", limit=None, max_len=35):
    pd = _pd()
    repo_root = find_repo_root(repo_root)
    text_util = load_text_util(repo_root)
    reference_mapping, _ = load_caption_sequences(repo_root, text_util)
    image_features = load_feature_map(repo_root)
    _, val_keys, test_keys = split_image_keys([k for k in reference_mapping if k in image_features])
    keys = test_keys if split == "test" else val_keys

    eval_path = Path(repo_root) / "artifacts" / "rnn" / "results" / "scratch_variation_eval.csv"
    if eval_path.exists():
        variation_eval = pd.read_csv(eval_path)
    else:
        variation_eval, _ = evaluate_all_variations(repo_root, split=split, limit=limit, max_len=max_len)

    best_rows = variation_eval.sort_values(["model_type", "scratch_bleu_4"], ascending=[True, False]).groupby("model_type").head(1)
    rows = []

    for best in best_rows.to_dict("records"):
        keras_model = make_keras_model(
            repo_root,
            text_util,
            best["model_type"],
            best["variation_name"],
            best["layers"],
            best["hidden_state"],
        )
        scratch_model = ImageCaptioningModel(keras_model, text_util, is_lstm=best["model_type"] == "LSTM")

        scratch_detail, scratch_elapsed, scratch_avg = evaluate_decoder(
            scratch_model,
            keys,
            image_features,
            reference_mapping,
            decoder=lambda m, feature, max_len: m.generate_caption(feature, max_len=max_len),
            max_len=max_len,
            limit=limit,
        )
        keras_detail, keras_elapsed, keras_avg = evaluate_decoder(
            keras_model,
            keys,
            image_features,
            reference_mapping,
            decoder=lambda m, feature, max_len: greedy_decode_keras(m, feature, text_util, max_len=max_len),
            max_len=max_len,
            limit=limit,
        )

        rows.extend(
            [
                _summary_row(best, "scratch", scratch_detail, scratch_elapsed, scratch_avg),
                _summary_row(best, "keras", keras_detail, keras_elapsed, keras_avg),
            ]
        )

    comparison = pd.DataFrame(rows)
    output_dir = Path(repo_root) / "artifacts" / "rnn" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "keras_vs_scratch.csv", index=False)
    return comparison


def max_length_sweep(repo_root=None, lengths=(10, 20, 35), split="test", limit=None):
    pd = _pd()
    repo_root = find_repo_root(repo_root)
    base = compare_best_keras_vs_scratch(repo_root, split=split, limit=limit, max_len=max(lengths))
    best = base.sort_values(["bleu_4", "avg_time_sec"], ascending=[False, True]).iloc[0]

    text_util = load_text_util(repo_root)
    reference_mapping, _ = load_caption_sequences(repo_root, text_util)
    image_features = load_feature_map(repo_root)
    _, val_keys, test_keys = split_image_keys([k for k in reference_mapping if k in image_features])
    keys = test_keys if split == "test" else val_keys

    keras_model = make_keras_model(
        repo_root,
        text_util,
        best["model_type"],
        best["variation_name"],
        best["layers"],
        best["hidden_state"],
    )
    model = ImageCaptioningModel(keras_model, text_util, is_lstm=best["model_type"] == "LSTM") if best["implementation"] == "scratch" else keras_model
    decoder = (
        (lambda m, feature, max_len: m.generate_caption(feature, max_len=max_len))
        if best["implementation"] == "scratch"
        else (lambda m, feature, max_len: greedy_decode_keras(m, feature, text_util, max_len=max_len))
    )

    rows = []
    for length in lengths:
        detail, elapsed, avg = evaluate_decoder(model, keys, image_features, reference_mapping, decoder, max_len=length, limit=limit)
        rows.append(
            {
                "model_type": best["model_type"],
                "variation_name": best["variation_name"],
                "implementation": best["implementation"],
                "max_len": length,
                "bleu_4": detail["bleu_4"].mean(),
                "meteor": detail["meteor"].mean(),
                "total_time_sec": elapsed,
                "avg_time_sec": avg,
            }
        )

    sweep = pd.DataFrame(rows)
    output_dir = Path(repo_root) / "artifacts" / "rnn" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(output_dir / "max_length_sweep.csv", index=False)
    return sweep


def qualitative_samples(repo_root=None, limit=None, n_per_bucket=4):
    pd = _pd()
    repo_root = find_repo_root(repo_root)
    detail_path = Path(repo_root) / "artifacts" / "rnn" / "results" / "scratch_caption_details.csv"
    if not detail_path.exists():
        _, details = evaluate_all_variations(repo_root, limit=limit)
    else:
        details = pd.read_csv(detail_path)

    text_util = load_text_util(repo_root)
    reference_mapping, _ = load_caption_sequences(repo_root, text_util)
    best_by_type = (
        details.groupby(["model_type", "variation_name"])["bleu_4"]
        .mean()
        .reset_index()
        .sort_values(["model_type", "bleu_4"], ascending=[True, False])
        .groupby("model_type")
        .head(1)
    )
    best_details = details.merge(best_by_type[["model_type", "variation_name"]], on=["model_type", "variation_name"])
    pivot = best_details.pivot_table(
        index="image_name",
        columns="model_type",
        values=["generated_caption", "bleu_4"],
        aggfunc="first",
    )
    pivot.columns = ["_".join(col).strip() for col in pivot.columns.values]
    pivot = pivot.reset_index()
    score_cols = [col for col in pivot.columns if col.startswith("bleu_4_")]
    pivot["avg_bleu_4"] = pivot[score_cols].mean(axis=1)

    median_order = (pivot["avg_bleu_4"] - pivot["avg_bleu_4"].median()).abs().sort_values().index
    buckets = [
        pivot.sort_values("avg_bleu_4", ascending=False).head(n_per_bucket),
        pivot.sort_values("avg_bleu_4").head(n_per_bucket),
        pivot.loc[median_order].head(n_per_bucket),
    ]
    samples = pd.concat(buckets, ignore_index=True).drop_duplicates("image_name").head(10)
    samples["ground_truth"] = samples["image_name"].map(lambda name: " | ".join(reference_mapping.get(name, [])[:5]))

    output_dir = Path(repo_root) / "artifacts" / "rnn" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    samples.to_csv(output_dir / "qualitative_samples.csv", index=False)
    return samples


def load_training_history(repo_root=None):
    pd = _pd()
    repo_root = find_repo_root(repo_root)
    history = pd.read_csv(Path(repo_root) / "src" / "rnn" / "hasil_variasi_model.csv")
    for column in ("history_loss", "history_val_loss"):
        history[column] = history[column].apply(ast.literal_eval)
    return history


def write_analysis_summary(repo_root=None):
    pd = _pd()
    repo_root = find_repo_root(repo_root)
    result_dir = Path(repo_root) / "artifacts" / "rnn" / "results"
    variation = pd.read_csv(result_dir / "scratch_variation_eval.csv") if (result_dir / "scratch_variation_eval.csv").exists() else pd.DataFrame()
    comparison = pd.read_csv(result_dir / "keras_vs_scratch.csv") if (result_dir / "keras_vs_scratch.csv").exists() else pd.DataFrame()
    max_len = pd.read_csv(result_dir / "max_length_sweep.csv") if (result_dir / "max_length_sweep.csv").exists() else pd.DataFrame()

    lines = ["# RNN/LSTM Experiment Summary", ""]
    if not variation.empty:
        best = variation.sort_values("scratch_bleu_4", ascending=False).iloc[0]
        lines.append(f"- Best scratch model: {best.model_type} {best.variation_name} with BLEU-4={best.scratch_bleu_4:.4f}.")
        lines.append("- Deeper/larger models should be judged with BLEU/METEOR and inference time, not loss alone.")
    if not comparison.empty:
        lines.append("- Keras vs scratch comparison is saved in `keras_vs_scratch.csv`; small score differences usually come from decoding and numerical details.")
    if not max_len.empty:
        best_len = max_len.sort_values("bleu_4", ascending=False).iloc[0]
        lines.append(f"- Best max caption length in sweep: {int(best_len.max_len)} with BLEU-4={best_len.bleu_4:.4f}.")
    lines.extend(
        [
            "- LSTM is expected to handle longer dependencies better than SimpleRNN because gates reduce vanishing-gradient effects.",
            "- Use `qualitative_samples.csv` for the 10-image high/medium/low qualitative comparison in the report.",
        ]
    )

    result_dir.mkdir(parents=True, exist_ok=True)
    path = result_dir / "analysis_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _summary_row(best, implementation, detail, elapsed, avg_time):
    return {
        "model_type": best["model_type"],
        "variation_name": best["variation_name"],
        "layers": int(best["layers"]),
        "hidden_state": int(best["hidden_state"]),
        "implementation": implementation,
        "bleu_4": detail["bleu_4"].mean(),
        "meteor": detail["meteor"].mean(),
        "total_time_sec": elapsed,
        "avg_time_sec": avg_time,
    }


def _pd():
    import pandas as pd

    return pd


if __name__ == "__main__":
    root = find_repo_root()
    variation_results, _ = evaluate_all_variations(root, limit=100)
    print(variation_results)
    print(compare_best_keras_vs_scratch(root, limit=100))
    print(max_length_sweep(root, limit=100))
    print(qualitative_samples(root, limit=100))
    print(write_analysis_summary(root))
