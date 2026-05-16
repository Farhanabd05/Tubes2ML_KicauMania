from dataclasses import dataclass
from pathlib import Path


ARCH_TAG = "preinject_v2"


@dataclass(frozen=True)
class RnnPaths:
    root: Path

    @classmethod
    def from_root(cls, root=None):
        return cls(find_repo_root(root))

    @property
    def dataset_dir(self):
        return self.root / "RNN_dataset"

    @property
    def image_dir(self):
        return self.dataset_dir / "Images"

    @property
    def captions_file(self):
        return self.dataset_dir / "captions.txt"

    @property
    def feature_dir(self):
        return self.root / "images_feature"

    @property
    def features_file(self):
        return self.feature_dir / "features.npy"

    @property
    def feature_names_file(self):
        return self.feature_dir / "image_names.npy"

    @property
    def module_dir(self):
        return self.root / "src" / "rnn"

    @property
    def vocab_dir(self):
        return self.module_dir / "output"

    @property
    def training_history_file(self):
        return self.results_dir / f"training_history_{ARCH_TAG}.csv"

    @property
    def weights_dir(self):
        return self.root / "artifacts" / "rnn" / "weights"

    @property
    def results_dir(self):
        return self.root / "artifacts" / "rnn" / "results"

    @property
    def feature_scaler_file(self):
        return self.root / "artifacts" / "rnn" / f"feature_scaler_{ARCH_TAG}.npz"

    def result_file(self, filename):
        return self.results_dir / filename

    def weight_file(self, model_type, layers, hidden_state):
        prefix = "LSTM" if model_type == "LSTM" else "SimpleRNN"
        layer_tag = {1: "Shallow", 2: "Deep", 3: "VeryDeep"}.get(int(layers), f"L{layers}")
        size_tag = "Small" if int(hidden_state) == 128 else "Mid" if int(hidden_state) == 256 else "Large"
        return self.weights_dir / f"{prefix}_PreInjectV2_{layer_tag}_{size_tag}_L{int(layers)}_H{int(hidden_state)}.weights.h5"


def find_repo_root(start=None):
    cwd = Path(start or Path.cwd()).resolve()
    for path in [cwd, *cwd.parents]:
        if (path / "src" / "rnn").exists() and (path / "RNN_dataset").exists():
            return path
    return cwd
