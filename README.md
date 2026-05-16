# Tubes2ML KicauMania

Implementasi Tugas Besar 2 IF3270 Pembelajaran Mesin untuk:

- CNN image classification dari scratch.
- RNN/LSTM image captioning pada Flickr8k dengan arsitektur pre-inject.

## Struktur Project

```text
src/
  cnn/      Implementasi layer dan model CNN from scratch
  rnn/      Pipeline RNN/LSTM image captioning
artifacts/
  rnn/
    results/    CSV dan ringkasan hasil evaluasi
    weights/    Bobot model RNN/LSTM, ditrack dengan Git LFS
images_feature/ Feature vector hasil ekstraksi CNN, generated lokal/Kaggle
RNN_dataset/    Dataset Flickr8k lokal, tidak perlu dipush
CNN_dataset/    Dataset CNN lokal, tidak perlu dipush
```

## Setup Lokal

1. Clone repository.

```bash
git clone https://github.com/Farhanabd05/Tubes2ML_KicauMania.git
cd Tubes2ML_KicauMania
```

2. Install dependency.

```bash
pip install -r requirements.txt
```

3. Jalankan notebook.

- CNN: Untuk menjalankan CNN, jalankan `cnn_notebook.ipynb` dari root repository dengan extension google Colab atau dengan link colab ini (https://colab.research.google.com/drive/1hI7YSPzZcR1KIbEYTwVGDTNYlPoZnqs5?usp=sharing)
- RNN/LSTM lokal: `src/rnn/rnn_notebook.ipynb`
- RNN/LSTM Kaggle GPU: `src/rnn/rnn_notebook_kaggle_gpu.ipynb`

## RNN/LSTM Artifact

Arsitektur RNN/LSTM yang dipakai saat ini adalah `preinject_v2`.

Bobot model disimpan di:

```text
artifacts/rnn/weights/*PreInjectV2*.weights.h5
```

Hasil evaluasi disimpan di:

```text
artifacts/rnn/results/
  training_history_preinject_v2.csv
  scratch_variation_eval.csv
  scratch_caption_details.csv
  keras_vs_scratch.csv
  max_length_sweep.csv
  qualitative_samples.csv
  analysis_summary.md
```

File weight besar ditrack menggunakan Git LFS. Setelah clone, jika weight belum terunduh, jalankan:

```bash
git lfs pull
```

## Menjalankan RNN/LSTM di Kaggle

Notebook Kaggle publik:

https://www.kaggle.com/code/farhanabd05/tubes-ml-2-v2

Dataset Flickr8k yang digunakan:

```text
Flickr 8k Dataset
adityajn105/flickr8k
```

### Input yang Dibutuhkan di Kaggle

Di Kaggle Notebook, tambahkan dua input:

1. **Project code zip**
   - Berisi kode project saja.
   - Wajib ada folder `src/`.
   - Tidak perlu memasukkan dataset, feature cache, atau artifact training.

2. **Flickr 8k Dataset**
   - Tambahkan langsung dari Kaggle lewat menu **Add Input**.
   - Cari dataset `adityajn105/flickr8k`.

Notebook akan mencari dataset Flickr8k dari `/kaggle/input`, lalu menyiapkan struktur `RNN_dataset/captions.txt` dan `RNN_dataset/Images` di `/kaggle/working`.

### File yang Tidak Perlu Masuk Zip

Jangan masukkan folder berikut ke zip code:

```text
.git/
RNN_dataset/
CNN_dataset/
images_feature/
artifacts/
__pycache__/
.ipynb_checkpoints/
```

Dataset Flickr8k sudah disediakan oleh Kaggle, sedangkan `images_feature/` dan `artifacts/` akan dibuat ulang oleh notebook jika diperlukan.

### Cara Membuat Zip Code untuk Kaggle

Jangan gunakan PowerShell `Compress-Archive`, karena pada beberapa kasus Kaggle menolak path di dalam zip yang memakai backslash seperti `src\rnn\modeling.py`.

Gunakan script Python berikut dari root repository agar path di dalam zip memakai forward slash `/`.

```powershell
@'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

root = Path.cwd()
zip_path = root / "tubes2ml-kicaumania-rnn-code-kaggle.zip"

include_roots = [
    root / "src",
    root / "README.md",
    root / "requirements.txt",
]

exclude_parts = {
    ".git",
    "__pycache__",
    ".ipynb_checkpoints",
    "artifacts",
    "images_feature",
    "RNN_dataset",
    "CNN_dataset",
}

files = []
for item in include_roots:
    if item.is_file():
        files.append(item)
    elif item.is_dir():
        for path in item.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(root)
            if any(part in exclude_parts for part in rel.parts):
                continue
            if path.suffix.lower() in {".pyc", ".pyo"}:
                continue
            files.append(path)

with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zf:
    for path in sorted(files):
        arcname = path.relative_to(root).as_posix()
        if "\\" in arcname:
            raise RuntimeError(f"Backslash ditemukan di archive name: {arcname}")
        zf.write(path, arcname)

with ZipFile(zip_path) as zf:
    bad = [name for name in zf.namelist() if "\\" in name]
    if bad:
        raise RuntimeError(f"Invalid zip entries: {bad[:10]}")
    print(zip_path)
    print(f"Total files: {len(zf.namelist())}")
    print("Backslash entries: 0")
'@ | python -
```

Jika command `python` tidak dikenali di Windows, jalankan script yang sama memakai Python path yang tersedia di mesin masing-masing.

### Cara Upload Zip ke Kaggle

1. Buka Kaggle.
2. Masuk ke menu **Datasets**.
3. Klik **New Dataset**.
4. Upload `tubes2ml-kicaumania-rnn-code-kaggle.zip`.
5. Beri nama dataset, misalnya:

```text
tubes2ml-kicaumania-rnn-code
```

6. Visibility boleh private atau public.
7. Klik **Create**.

### Cara Setup Notebook Kaggle

1. Buka notebook Kaggle:

```text
https://www.kaggle.com/code/farhanabd05/tubes-ml-2-v2
```

2. Klik **Copy & Edit** jika ingin menjalankan di akun sendiri.
3. Di panel kanan, set:

```text
Accelerator: GPU T4 x2
Internet: On
```

Internet diperlukan jika notebook harus mengunduh pretrained InceptionV3 ImageNet weights atau package/resource yang belum tersedia.

4. Klik **Add Input** lalu tambahkan:

```text
tubes2ml-kicaumania-rnn-code
adityajn105/flickr8k
```

5. Jalankan notebook dari atas ke bawah.

### Konfigurasi Penting Notebook Kaggle

Konfigurasi utama ada di cell awal notebook:

```python
SEQ_LEN = 35
EPOCHS = 5
EVAL_LIMIT = 100
PER_REPLICA_IMAGE_BATCH = 32
RUN_FULL_TRAINING = True
RETRAIN_EXISTING_WEIGHTS = False
USE_TF_DATASET = True
```

Catatan:

- Untuk GPU T4 x2, `PER_REPLICA_IMAGE_BATCH = 32` berarti total batch efektif 64.
- Jika terkena out-of-memory, turunkan `PER_REPLICA_IMAGE_BATCH` menjadi 16.
- Jika hanya ingin smoke test cepat, ubah `EPOCHS = 1`.
- Jika weight sudah ada dan tidak ingin training ulang, pastikan `RETRAIN_EXISTING_WEIGHTS = False`.

### Output Kaggle

Setelah notebook selesai, output akan dibuat di:

```text
/kaggle/working/rnn_preinject_v2_outputs.zip
```

Zip tersebut berisi weight `PreInjectV2`, scaler, dan CSV hasil evaluasi. Download file ini dari panel **Output** Kaggle jika ingin dipindahkan ke lokal.

## Kelompok KicauMania

| NIM      | Nama                         | Pembagian Tugas |
| :------: | :--------------------------- | :-------------- |
| 13523023 | Muhammad Aufa Farabi         | Implementasi CNN |
| 13523042 | Abdullah Farhan              | Implementasi RNN LSTM (Bagian 0 dan 4 sampai 6), Fix Bug|
| 13523051 | Ferdinand Gabe Tua Sinaga    | Implementasi dan analisis RNN (Bagian 1 sampai 3) |
