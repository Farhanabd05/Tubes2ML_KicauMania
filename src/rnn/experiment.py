import ast
import time

import numpy as np

from .ImageCaptioningScratch import ImageCaptioningModel
from .modeling import build_caption_model, greedy_decode_keras
from .paths import ARCH_TAG, RnnPaths, find_repo_root
from .utils.text_utils import CaptionPreprocessor


VARIATIONS = [
    {"variation_name": "Shallow_Small", "layers": 1, "hidden_state": 128},
    {"variation_name": "Deep_Small", "layers": 2, "hidden_state": 128},
    {"variation_name": "VeryDeep_Small", "layers": 3, "hidden_state": 128},
    {"variation_name": "Shallow_Mid", "layers": 1, "hidden_state": 256},
    {"variation_name": "Shallow_Large", "layers": 1, "hidden_state": 512},
    {"variation_name": "Deep_Large", "layers": 2, "hidden_state": 512},
]


def load_feature_map(repo_root):
    paths = RnnPaths.from_root(repo_root)
    features = np.load(paths.features_file)
    names = np.load(paths.feature_names_file, allow_pickle=True)
    return {str(name): features[idx] for idx, name in enumerate(names)}


def fit_feature_scaler(image_features, train_keys):
    train_features = np.asarray([image_features[key] for key in train_keys], dtype=np.float32)
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def save_feature_scaler(repo_root, mean, std):
    paths = RnnPaths.from_root(repo_root)
    paths.feature_scaler_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(paths.feature_scaler_file, architecture=ARCH_TAG, mean=mean, std=std)


def load_feature_scaler(repo_root):
    paths = RnnPaths.from_root(repo_root)
    if not paths.feature_scaler_file.exists():
        raise FileNotFoundError(f"Feature scaler {ARCH_TAG} belum ada: {paths.feature_scaler_file}")

    data = np.load(paths.feature_scaler_file, allow_pickle=True)
    architecture = str(data["architecture"]) if "architecture" in data else ""
    if architecture != ARCH_TAG:
        raise ValueError(f"Feature scaler bukan {ARCH_TAG}: {paths.feature_scaler_file}")
    return data["mean"].astype(np.float32), data["std"].astype(np.float32)


def load_or_create_feature_scaler(repo_root, image_features, train_keys, force=False):
    paths = RnnPaths.from_root(repo_root)
    if paths.feature_scaler_file.exists() and not force:
        return load_feature_scaler(paths.root)

    mean, std = fit_feature_scaler(image_features, train_keys)
    save_feature_scaler(paths.root, mean, std)
    return mean, std


def scale_feature_map(image_features, mean, std):
    return {
        key: ((np.asarray(feature, dtype=np.float32) - mean) / std).astype(np.float32)
        for key, feature in image_features.items()
    }


def load_text_util(repo_root, sequence_length=35, force_build=False):
    paths = RnnPaths.from_root(repo_root)
    text_util = CaptionPreprocessor(sequence_length=sequence_length)
    text_util.build_vocabulary(
        str(paths.captions_file),
        force_build=force_build,
        directory=str(paths.vocab_dir),
    )
    return text_util


def load_caption_sequences(repo_root, text_util):
    paths = RnnPaths.from_root(repo_root)
    mapping = text_util.get_image_to_captions_mapping(str(paths.captions_file))
    sequence_mapping = {}
    for image_name, captions in mapping.items():
        sequence_mapping[image_name] = text_util.pad_sequences(text_util.texts_to_sequences(captions))
    return mapping, sequence_mapping


def split_image_keys(keys, train_size=6000, val_size=1000, seed=42):
    keys = sorted(keys)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    return keys[:train_size], keys[train_size:train_size + val_size], keys[train_size + val_size:]


def weight_path(repo_root, model_type, variation_name, layers, hidden_state):
    return RnnPaths.from_root(repo_root).weight_file(model_type, layers, hidden_state)


def make_keras_model(repo_root, text_util, model_type, variation_name, layers, hidden_state):
    model = build_caption_model(
        is_lstm=model_type == "LSTM",
        layers=int(layers),
        hidden_state=int(hidden_state),
        vocab_size=text_util.vocab_size,
        sequence_length=text_util.sequence_length,
    )
    path = weight_path(repo_root, model_type, variation_name, layers, hidden_state)
    if not path.exists():
        raise FileNotFoundError(f"Bobot {ARCH_TAG} belum ada: {path}")
    model.load_weights(str(path))
    return model


def score_caption(references, hypothesis):
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from nltk.translate.meteor_score import meteor_score

    ref_tokens = [clean_generated_caption(ref).split() for ref in references]
    hyp_tokens = clean_generated_caption(hypothesis).split()
    smoothing = SmoothingFunction().method1
    bleu_4 = sentence_bleu(ref_tokens, hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    try:
        meteor = meteor_score(ref_tokens, hyp_tokens)
    except LookupError:
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


def load_evaluation_context(repo_root=None, split="test", train_size=6000, val_size=1000, seed=42):
    paths = RnnPaths.from_root(repo_root)
    text_util = load_text_util(paths.root)
    reference_mapping, _ = load_caption_sequences(paths.root, text_util)
    raw_image_features = load_feature_map(paths.root)
    train_keys, val_keys, test_keys = split_image_keys(
        [key for key in reference_mapping if key in raw_image_features],
        train_size=train_size,
        val_size=val_size,
        seed=seed,
    )
    mean, std = load_or_create_feature_scaler(paths.root, raw_image_features, train_keys)
    image_features = scale_feature_map(raw_image_features, mean, std)

    if split == "train":
        keys = train_keys
    elif split == "val":
        keys = val_keys
    elif split == "test":
        keys = test_keys
    else:
        raise ValueError("split harus salah satu dari: train, val, test.")

    return text_util, reference_mapping, image_features, keys, (train_keys, val_keys, test_keys)


def evaluate_all_variations(repo_root=None, split="test", limit=None, max_len=35):
    pd = _pd()
    paths = RnnPaths.from_root(repo_root)
    text_util, reference_mapping, image_features, keys, _ = load_evaluation_context(paths.root, split=split)

    history = load_training_history(paths.root)
    result_rows = []
    detail_frames = []

    for row in history.to_dict("records"):
        model = make_keras_model(
            paths.root,
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
        detail.insert(0, "architecture", ARCH_TAG)
        detail.insert(1, "model_type", row["model_type"])
        detail.insert(2, "variation_name", row["variation_name"])
        detail_frames.append(detail)
        caption_counts = detail["generated_caption"].value_counts()
        top_caption_frequency = caption_counts.iloc[0] / len(detail) if len(detail) else 0.0

        result_rows.append(
            {
                "architecture": ARCH_TAG,
                "model_type": row["model_type"],
                "variation_name": row["variation_name"],
                "layers": int(row["layers"]),
                "hidden_state": int(row["hidden_state"]),
                "scratch_bleu_4": detail["bleu_4"].mean(),
                "scratch_meteor": detail["meteor"].mean(),
                "scratch_total_time_sec": elapsed,
                "scratch_avg_time_sec": avg_time,
                "unique_captions": int(detail["generated_caption"].nunique()),
                "top_caption_frequency": float(top_caption_frequency),
                "training_time_sec": row.get("training_time_sec"),
                "final_loss": row.get("final_loss"),
                "final_val_loss": row.get("final_val_loss"),
            }
        )

    results = pd.DataFrame(result_rows)
    details = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(paths.result_file("scratch_variation_eval.csv"), index=False)
    details.to_csv(paths.result_file("scratch_caption_details.csv"), index=False)
    return results, details


def refresh_scratch_caption_artifacts(repo_root=None):
    pd = _pd()
    paths = RnnPaths.from_root(repo_root)
    detail_path = paths.result_file("scratch_caption_details.csv")
    if not detail_path.exists():
        raise FileNotFoundError(f"Caption detail artifact tidak ditemukan: {detail_path}")

    text_util = load_text_util(paths.root)
    reference_mapping, _ = load_caption_sequences(paths.root, text_util)
    details = pd.read_csv(detail_path)
    _assert_current_arch(details, detail_path)
    required_columns = {"architecture", "model_type", "variation_name", "image_name", "generated_caption"}
    missing_columns = required_columns.difference(details.columns)
    if missing_columns:
        raise ValueError(f"Kolom wajib tidak ada di {detail_path}: {sorted(missing_columns)}")

    details["generated_caption"] = details["generated_caption"].map(clean_generated_caption)
    scores = details.apply(
        lambda row: score_caption(reference_mapping[row["image_name"]], row["generated_caption"]),
        axis=1,
    )
    details["bleu_4"] = [score[0] for score in scores]
    details["meteor"] = [score[1] for score in scores]
    details.to_csv(detail_path, index=False)

    metrics = (
        details.groupby(["model_type", "variation_name"], as_index=False)
        .agg(scratch_bleu_4=("bleu_4", "mean"), scratch_meteor=("meteor", "mean"))
    )
    diversity = (
        details.groupby(["model_type", "variation_name"])["generated_caption"]
        .agg(unique_captions="nunique", top_caption_frequency=lambda values: values.value_counts(normalize=True).iloc[0])
        .reset_index()
    )
    metrics = metrics.merge(diversity, on=["model_type", "variation_name"], how="left")
    metrics.insert(0, "architecture", ARCH_TAG)

    summary_path = paths.result_file("scratch_variation_eval.csv")
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        _assert_current_arch(summary, summary_path)
        summary = summary.drop(columns=["scratch_bleu_4", "scratch_meteor", "unique_captions", "top_caption_frequency"], errors="ignore")
        summary = summary.merge(metrics, on=["architecture", "model_type", "variation_name"], how="left")
        ordered_columns = [
            "architecture",
            "model_type",
            "variation_name",
            "layers",
            "hidden_state",
            "scratch_bleu_4",
            "scratch_meteor",
            "scratch_total_time_sec",
            "scratch_avg_time_sec",
            "unique_captions",
            "top_caption_frequency",
            "training_time_sec",
            "final_loss",
            "final_val_loss",
        ]
        summary = summary[[column for column in ordered_columns if column in summary.columns]]
    else:
        history = load_training_history(paths.root)
        summary = history.merge(metrics, on=["architecture", "model_type", "variation_name"], how="left")

    summary.to_csv(summary_path, index=False)
    return summary, details


def compare_best_keras_vs_scratch(repo_root=None, split="test", limit=None, max_len=35):
    pd = _pd()
    paths = RnnPaths.from_root(repo_root)
    text_util, reference_mapping, image_features, keys, _ = load_evaluation_context(paths.root, split=split)

    eval_path = paths.result_file("scratch_variation_eval.csv")
    if eval_path.exists():
        variation_eval = pd.read_csv(eval_path)
        _assert_current_arch(variation_eval, eval_path)
    else:
        variation_eval, _ = evaluate_all_variations(paths.root, split=split, limit=limit, max_len=max_len)

    best_rows = variation_eval.sort_values(["model_type", "scratch_bleu_4"], ascending=[True, False]).groupby("model_type").head(1)
    rows = []

    for best in best_rows.to_dict("records"):
        keras_model = make_keras_model(
            paths.root,
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
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(paths.result_file("keras_vs_scratch.csv"), index=False)
    return comparison


def max_length_sweep(repo_root=None, lengths=(10, 20, 35), split="test", limit=None):
    pd = _pd()
    paths = RnnPaths.from_root(repo_root)
    base = compare_best_keras_vs_scratch(paths.root, split=split, limit=limit, max_len=max(lengths))
    best = base.sort_values(["bleu_4", "avg_time_sec"], ascending=[False, True]).iloc[0]

    text_util, reference_mapping, image_features, keys, _ = load_evaluation_context(paths.root, split=split)

    keras_model = make_keras_model(
        paths.root,
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
        caption_counts = detail["generated_caption"].value_counts()
        top_caption_frequency = caption_counts.iloc[0] / len(detail) if len(detail) else 0.0
        rows.append(
            {
                "architecture": ARCH_TAG,
                "model_type": best["model_type"],
                "variation_name": best["variation_name"],
                "implementation": best["implementation"],
                "max_len": length,
                "bleu_4": detail["bleu_4"].mean(),
                "meteor": detail["meteor"].mean(),
                "total_time_sec": elapsed,
                "avg_time_sec": avg,
                "unique_captions": int(detail["generated_caption"].nunique()),
                "top_caption_frequency": float(top_caption_frequency),
            }
        )

    sweep = pd.DataFrame(rows)
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(paths.result_file("max_length_sweep.csv"), index=False)
    return sweep


def qualitative_samples(repo_root=None, limit=None, n_per_bucket=4):
    pd = _pd()
    paths = RnnPaths.from_root(repo_root)
    detail_path = paths.result_file("scratch_caption_details.csv")
    if not detail_path.exists():
        _, details = evaluate_all_variations(paths.root, limit=limit)
    else:
        details = pd.read_csv(detail_path)
        _assert_current_arch(details, detail_path)

    text_util = load_text_util(paths.root)
    reference_mapping, _ = load_caption_sequences(paths.root, text_util)
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
    samples.insert(0, "architecture", ARCH_TAG)
    samples["ground_truth"] = samples["image_name"].map(lambda name: " | ".join(reference_mapping.get(name, [])[:5]))

    paths.results_dir.mkdir(parents=True, exist_ok=True)
    samples.to_csv(paths.result_file("qualitative_samples.csv"), index=False)
    return samples


def assert_anti_collapse_acceptance(repo_root=None, min_unique=20, max_top_caption_frequency=0.25):
    pd = _pd()
    paths = RnnPaths.from_root(repo_root)
    results = _read_arch_csv_if_exists(paths.result_file("scratch_variation_eval.csv"), pd)
    if results.empty:
        raise FileNotFoundError("scratch_variation_eval.csv belum ada untuk acceptance check.")

    best_by_type = (
        results.sort_values(["model_type", "scratch_bleu_4"], ascending=[True, False])
        .groupby("model_type")
        .head(1)
        .copy()
    )
    failing = best_by_type[
        (best_by_type["unique_captions"] < min_unique)
        | (best_by_type["top_caption_frequency"] > max_top_caption_frequency)
    ]
    if not failing.empty:
        details = failing[
            ["model_type", "variation_name", "unique_captions", "top_caption_frequency"]
        ].to_dict("records")
        raise AssertionError(f"Anti-collapse acceptance gagal untuk {ARCH_TAG}: {details}")

    return best_by_type


def load_training_history(repo_root=None):
    pd = _pd()
    paths = RnnPaths.from_root(repo_root)
    history = pd.read_csv(paths.training_history_file)
    _assert_current_arch(history, paths.training_history_file)
    for column in ("history_loss", "history_val_loss"):
        history[column] = history[column].apply(ast.literal_eval)
    return history


def write_analysis_summary(repo_root=None):
    pd = _pd()
    paths = RnnPaths.from_root(repo_root)
    variation_path = paths.result_file("scratch_variation_eval.csv")
    comparison_path = paths.result_file("keras_vs_scratch.csv")
    max_len_path = paths.result_file("max_length_sweep.csv")
    variation = _read_arch_csv_if_exists(variation_path, pd)
    comparison = _read_arch_csv_if_exists(comparison_path, pd)
    max_len = _read_arch_csv_if_exists(max_len_path, pd)

    lines = ["# RNN/LSTM Experiment Summary", ""]
    lines.append(f"- Architecture: `{ARCH_TAG}`.")
    lines.append("- Model follows the assignment pre-inject design: `Image_Projection` is inserted as timestep `t=-1` before `<start>`, then `Drop_Image_Timestep` aligns recurrent outputs with shifted caption targets.")
    if not variation.empty:
        best = variation.sort_values("scratch_bleu_4", ascending=False).iloc[0]
        lines.append(f"- Best scratch model: {best.model_type} {best.variation_name} with BLEU-4={best.scratch_bleu_4:.4f}.")
        if "unique_captions" in variation.columns and "top_caption_frequency" in variation.columns:
            lines.append(
                f"- Best model diversity: {int(best.unique_captions)} unique captions; top-caption frequency={best.top_caption_frequency:.2%}."
            )
        lines.append("- Deeper/larger models should be judged with BLEU/METEOR and inference time, not loss alone.")
    if not comparison.empty:
        lines.append("- Keras vs scratch comparison is saved in `keras_vs_scratch.csv`; small score differences usually come from decoding and numerical details.")
    if not max_len.empty:
        best_len = max_len.sort_values("bleu_4", ascending=False).iloc[0]
        lines.append(f"- Best max caption length in sweep: {int(best_len.max_len)} with BLEU-4={best_len.bleu_4:.4f}.")
    lines.extend(
        [
            "- Preinject-v2 smoke checks cover the 34-step caption input/output shape, t=-1 image timestep, and sample-weight pad masking.",
            "- LSTM is expected to handle longer dependencies better than SimpleRNN because gates reduce vanishing-gradient effects.",
            "- Use `qualitative_samples.csv` for the 10-image high/medium/low qualitative comparison in the report.",
        ]
    )

    paths.results_dir.mkdir(parents=True, exist_ok=True)
    path = paths.result_file("analysis_summary.md")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _summary_row(best, implementation, detail, elapsed, avg_time):
    caption_counts = detail["generated_caption"].value_counts()
    top_caption_frequency = caption_counts.iloc[0] / len(detail) if len(detail) else 0.0
    return {
        "architecture": ARCH_TAG,
        "model_type": best["model_type"],
        "variation_name": best["variation_name"],
        "layers": int(best["layers"]),
        "hidden_state": int(best["hidden_state"]),
        "implementation": implementation,
        "bleu_4": detail["bleu_4"].mean(),
        "meteor": detail["meteor"].mean(),
        "total_time_sec": elapsed,
        "avg_time_sec": avg_time,
        "unique_captions": int(detail["generated_caption"].nunique()),
        "top_caption_frequency": float(top_caption_frequency),
    }


def _pd():
    import pandas as pd

    return pd


def _assert_current_arch(frame, path):
    if "architecture" not in frame.columns:
        raise ValueError(f"{path} adalah artifact lama tanpa kolom architecture; regenerate {ARCH_TAG}.")
    invalid = set(frame["architecture"].dropna().astype(str)) - {ARCH_TAG}
    if invalid:
        raise ValueError(f"{path} bukan artifact {ARCH_TAG}: {sorted(invalid)}")


def _read_arch_csv_if_exists(path, pd):
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    _assert_current_arch(frame, path)
    return frame


if __name__ == "__main__":
    root = find_repo_root()
    variation_results, _ = evaluate_all_variations(root, limit=100)
    print(variation_results)
    print(compare_best_keras_vs_scratch(root, limit=100))
    print(max_length_sweep(root, limit=100))
    print(qualitative_samples(root, limit=100))
    print(write_analysis_summary(root))
