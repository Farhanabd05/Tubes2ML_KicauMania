# Tubes2ML KicauMania

Implementasi Tugas Besar 2 IF3270 Pembelajaran Mesin untuk CNN image classification dan RNN/LSTM image captioning.

## Struktur

```text
src/
  cnn/      Implementasi layer dan model CNN from scratch
  rnn/      Implementasi decoder RNN/LSTM dan pipeline captioning
  utils/    Utility image loading dan feature extraction
CNN_dataset/
RNN_dataset/
images_feature/
```

## Setup

```bash
pip install -r requirements.txt
```

## Menjalankan Notebook

- CNN: jalankan `cnn_notebook.ipynb` dari root repository.
- RNN/LSTM: jalankan `src/rnn/rnn_notebook.ipynb` dari folder `src/rnn`.

## Catatan Artifact

Bobot RNN/LSTM saat ini berada di `src/weights/`. Dari `src/rnn/rnn_notebook.ipynb`, path relatifnya adalah `../weights/...`.
