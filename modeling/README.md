# Pemodelan

| Berkas | Isi |
|---|---|
| `pemodelan.py` | Fungsi bersama: pemuat data dengan pagar data uji, tiga jaringan, pelatihan, metrik, log run |
| `01_baseline.ipynb` | Pagar data uji, tujuh pembanding di bagian validasi, perkiraan waktu pencarian |
| `02_pemilihan.ipynb` | Tahap A, tahap B, dan ablasi; hanya bagian validasi (belum dibuat) |
| `03_evaluasi_uji.ipynb` | Evaluasi bagian uji, sekali jalan, membaca `konfigurasi_final.json` (belum dibuat) |
| `hasil/log_run.csv` | Satu baris per run: perangkat, seed, konfigurasi, epoch terbaik, waktu latih, metrik validasi |
| `hasil/prediksi_val/` | Prediksi validasi dan kurva loss per run |

Bagian uji hanya bisa dibuka lewat `muat_uji`, yang menolak berjalan sebelum
`konfigurasi_final.json` ditulis dan dikunci oleh notebook 02.

Run dicatat bersama penanda perangkat (`mesin`). Hasil numerik CPU dan GPU
sedikit berbeda, jadi semua run satu tahap harus berasal dari perangkat yang sama;
run dari perangkat lain tidak dipakai ulang.

## Menjalankan di Ubuntu dengan GPU NVIDIA

Petunjuk ini belum diuji di GPU. Pada komputer dengan dua GPU, pakai satu saja
(yang lebih baru) supaya semua run berasal dari perangkat yang sama.

```bash
# 1. driver NVIDIA (sekali saja; versi 550 ke atas), lalu reboot
sudo ubuntu-drivers install
nvidia-smi                        # catat indeks GPU yang dipakai, misalnya RTX 4060

# 2. salin repo dan buat lingkungan Python 3.11–3.13 (Ubuntu 24.04: bawaan 3.12; 22.04 terlalu lama)
sudo apt install -y git python3-venv
git clone https://github.com/bernardswidiya/cnn-lstm-mcdropout-streamflow.git
cd cnn-lstm-mcdropout-streamflow
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
sed 's/^tensorflow==/tensorflow[and-cuda]==/' modeling/requirements.txt > /tmp/req.txt
.venv/bin/python -m pip install -r /tmp/req.txt

# 3. pilih satu GPU (ganti 0 dengan indeks dari nvidia-smi)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
.venv/bin/python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# 4. jalankan notebook
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 modeling/01_baseline.ipynb
```

Langkah 3 harus dijalankan ulang di setiap terminal baru, atau ditulis ke
`~/.bashrc`. Pemakaian GPU bisa dipantau dengan `watch -n 2 nvidia-smi`.
