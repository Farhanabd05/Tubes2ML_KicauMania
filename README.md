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

## Cara Menjalankan
### Cara
1. Clone repository ini
```bash
git clone https://github.com/Farhanabd05/Tubes1ML-MobileLegend.git
```
2. pip install -r requirements.txt
3. Untuk menjalankan CNN, jalankan `cnn_notebook.ipynb` dari root repository dengan extension google Colab atau dengan link colab ini (https://colab.research.google.com/drive/1hI7YSPzZcR1KIbEYTwVGDTNYlPoZnqs5?usp=sharing) dan untuk menjalankan RNN/LSTM, jalankan `src/rnn/rnn_notebook.ipynb` dari folder `src/rnn`.

## Kelompok 40 - MobileLegend
|   NIM    |                  Nama                  | Pembagian Tugas                    |
| :------: | :------------------------------------: | :--------------------------------: |
| 13523023 |           Muhammad Aufa Farabi         | Implementasi CNN|
| 13523042 |              Abdullah Farhan           | Implementasi     |
| 13523051 |      Ferdinand Gabe Tua Sinaga         | Implementasi  |

## Catatan Artifact

Bobot RNN/LSTM saat ini berada di `src/weights/`. Dari `src/rnn/rnn_notebook.ipynb`, path relatifnya adalah `../weights/...`.
