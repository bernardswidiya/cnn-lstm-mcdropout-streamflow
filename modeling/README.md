# Pemodelan

| Berkas | Isi |
|---|---|
| `pemodelan.py` | Fungsi bersama: pemuat data dengan pagar data uji, tiga jaringan, pelatihan, metrik, log run |
| `01_baseline.ipynb` | Pagar data uji, tujuh pembanding di bagian validasi, perkiraan waktu pencarian |
| `02_pemilihan.ipynb` | Tahap A, tahap B, tahap R (hujan v2), dan ablasi; hanya bagian validasi |
| `pekerja.py` | Proses pekerja untuk pelatihan paralel (`jalankan_paralel` di `pemodelan.py`) |
| `03_evaluasi.ipynb` | Evaluasi akhir dari bobot tersimpan (tanpa latih ulang). `BAGIAN = "validasi"` di sel pertama (sekarang); `BAGIAN = "uji"` hanya sekali, dengan `BUKA_UJI=1`, bila `hasil/uji_dibuka.txt` belum ada dan `konfigurasi_final.json` sudah dikunci. Harus dijalankan di perangkat run tercatat (RTX 4060) |
| `hasil/evaluasi_<bagian>_*.csv`, `figur/eval_<bagian>_*.png` | Keluaran notebook 03: per run, tabel A–C, pemeriksaan terhadap tahap D, konfigurasi dan hash-nya |
| `hasil/log_run.csv` | Satu baris per run: perangkat, seed, konfigurasi, epoch terbaik, waktu latih, metrik validasi |
| `hasil/prediksi_val/` | Prediksi validasi dan kurva loss per run |
| `hasil/tahap_a_*.csv`, `hasil/pilihan_tahap_a.json` | Hasil tahap A per run, ringkasan per pilihan, dan pilihan yang dikunci |
| `hasil/tahap_r_*.csv`, `hasil/pilihan_tahap_r.json` | Hasil tahap R (varian hujan v2 R0–R5) per run, ringkasan, dan varian terpilih |
| `hasil/tahap_r3_final_*.csv`, `hasil/tahap_r3_regresi_linear.csv` | Model terpilih tahap B dengan data R3_titik (8 seed, MC Dropout) dan regresi linear |
| `hasil/ablasi_data_*.csv` | Ablasi kontribusi data CNN-LSTM T21: debit, + hujan, + suhu, + atribut |
| `hasil/ablasi_atribut_acak_*.csv` | Kontrol atribut acak [Li2022] untuk CNN-LSTM T21 |
| `hasil/perbandingan_final_validasi.csv` | Alur utama: CNN-LSTM T21 (R3_titik) dan ARIMA, persistensi sebagai acuan; S_LSTM, LSTM target, dan regresi linear hanya lampiran |

Pilihan data tahap A yang tensornya tidak ada di `processed/` (jendela selain 14
hari, hujan GHCN, log10(√x + 0,1)) disusun di memori oleh `tensor_pilihan`: deret
harian direkonstruksi dari sampel latih dan validasi tensor yang tersimpan, lalu
disusun dengan fungsi `susun` dari preprocessing tahap 24. Untuk pilihan dasar,
hasilnya identik dengan file di `processed/`. Hari bagian uji tidak dimuat.

Pelatihan paralel: `jalankan_paralel` membagi run ke beberapa proses yang berbagi
satu GPU. Setiap proses menulis ke `hasil/log_run_p{k}.csv`, lalu semua log
digabung ke `log_run.csv` dengan pemeriksaan tidak ada baris ganda. Keluaran
pekerja ada di `cache/pekerja/`.

Bagian uji hanya bisa dibuka lewat `muat_uji`, yang menolak berjalan sebelum
`konfigurasi_final.json` ditulis dan dikunci oleh notebook 02.

Run dicatat bersama penanda perangkat (`mesin`). Hasil numerik CPU dan GPU
sedikit berbeda, jadi semua run satu tahap harus berasal dari perangkat yang sama;
run dari perangkat lain tidak dipakai ulang.

## Menjalankan di Ubuntu dengan GPU NVIDIA

Sudah diuji di Ubuntu 24.04 dengan NVIDIA GeForce RTX 4060 (driver 535, CUDA 12.2),
Python 3.12, TensorFlow 2.20.0 dan Keras 3.15.1. Driver 535 lebih lama dari yang
disarankan, tetapi berjalan; peringatan `cuStreamGetGreenCtx` dan `layout failed`
di stderr tidak berpengaruh. Determinisme per seed terverifikasi: dua run
CNN-LSTM dengan seed yang sama menghasilkan val_loss, loss latih, dan prediksi
yang identik sampai tingkat bit, juga ketika 2–3 proses berbagi GPU.

Pada komputer dengan dua GPU, pakai satu saja (yang lebih baru) supaya semua run
berasal dari perangkat yang sama. Di komputer uji, RTX 4060 berindeks 0 dan
GTX 1050 Ti berindeks 1 (urutan PCI).

```bash
# 1. driver NVIDIA (sekali saja; versi 550 ke atas), lalu reboot
sudo ubuntu-drivers install
nvidia-smi                        # catat indeks GPU yang dipakai, misalnya RTX 4060

# 2. salin repo dan buat lingkungan Python 3.11–3.13 (Ubuntu 24.04: bawaan 3.12; 22.04 terlalu lama)
sudo apt install -y git python3-venv   # tanpa sudo: python3 -m venv --without-pip, lalu pasang pip dari wheel
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
`~/.bashrc`. Proses pekerja mewarisi variabel ini dari notebook. Pemakaian GPU
bisa dipantau dengan `watch -n 2 nvidia-smi`.

Waktu di RTX 4060: satu run CNN-LSTM pengaturan tengah sekitar 20 detik (0,21
detik per epoch), sekitar 3 kali lebih cepat per epoch daripada CPU Apple M3
Pro. Dua proses yang berbagi GPU memberi throughput 1,5–1,7 kali satu proses;
proses ketiga hampir tidak menambah throughput untuk CNN-LSTM besar.

Tensor hujan v2 (`processed/v2/`) dimuat dengan `muat_latih_val(h, "V17", 8, "v2_<kode>")`, misalnya
`"v2_R3_das"`; kodenya ada di `processed/v2/varian_v2.json`. Varian R0 identik dengan pilihan tahap A
(`hujan="ghcn_idw"`), sehingga run R0 diambil dari log tahap A.
