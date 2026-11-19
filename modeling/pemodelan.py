"""Fungsi bersama notebook pemodelan 01, 02, dan 03.

Isi: pemuat data dengan pagar data uji, tiga arsitektur jaringan (LSTM target,
S_LSTM ala Le dkk. 2025, CNN-LSTM), pelatihan dengan early stopping, metrik,
dan pencatatan setiap run ke modeling/hasil/log_run.csv.

Pagar data uji: notebook 01 dan 02 hanya boleh memanggil `muat_latih_val` dan
`muat_deret_target`. Keduanya membuang bagian uji sebelum mengembalikan data
dan gagal (AssertionError) bila ada indeks uji yang ikut termuat. Bagian uji
hanya dapat dibuka lewat `muat_uji`, yang menolak berjalan sebelum
modeling/konfigurasi_final.json ditulis dan dikunci oleh notebook 02.
"""
import json, os, platform, time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# --- lokasi folder: dapat dipanggil dari akar repo maupun dari folder modeling/
ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "processed"
MODELING = ROOT / "modeling"
HASIL = MODELING / "hasil"
PREDIKSI = HASIL / "prediksi_val"
FIGUR = ROOT / "figur"
LOG_RUN = HASIL / "log_run.csv"
KONFIG_FINAL = MODELING / "konfigurasi_final.json"
for _d in (HASIL, PREDIKSI, FIGUR):
    _d.mkdir(parents=True, exist_ok=True)

META = json.loads((PROC / "metadata.json").read_text())
N_HARI = int(META["n_hari"])
I_LATIH, I_VAL = int(N_HARI * 0.70), int(N_HARI * 0.85)   # batas 70:15:15, sama dengan preprocessing tahap 22
PERIODE = pd.date_range(META["periode"][0], META["periode"][1], freq="D")
assert len(PERIODE) == N_HARI
TARGET = META["pos_target"]
HORIZON = list(META["horizon"])
N_DIN = len(META["channel_dinamis"])

VARIAN_FOLDER = {"V17": PROC, "V18": PROC / "varian" / "V18", "V21": PROC / "varian" / "V21",
                 "V17_daymet_asli": PROC / "varian" / "V17_daymet_asli"}


# ---------------------------------------------------------------------------
# Pagar data uji
# ---------------------------------------------------------------------------
def pagar_uji(idx, bagian):
    """Gagal bila ada sampel uji (kode bagian 2 atau 3, atau indeks target >= awal bagian uji)."""
    idx, bagian = np.asarray(idx), np.asarray(bagian)
    assert np.isin(bagian, [0, 1]).all(), "PAGAR: sampel uji (kode bagian 2/3) ikut termuat"
    assert idx.max() < I_VAL, f"PAGAR: indeks target {idx.max()} >= awal bagian uji {I_VAL}"
    return True


def muat_latih_val(h, varian="V17"):
    """Tensor horizon h untuk bagian latih dan validasi saja.

    Mengembalikan dict berisi X_latih, y_latih, idx_latih, X_val, y_val, idx_val,
    i_target, pos, kanal, skala debit pos target (rata, std), dan tanggal target.
    Bagian uji dibuang di dalam fungsi ini, sebelum apa pun dikembalikan."""
    if varian == "V17_daymet_asli":
        raise ValueError("V17_daymet_asli tidak ikut seleksi (kebocoran ±7 jam); hanya untuk notebook 03")
    with np.load(VARIAN_FOLDER[varian] / f"tensor_h{h}.npz", allow_pickle=True) as d:
        bagian = d["bagian"]
        pilih = bagian <= 1
        X, y, idx = d["X"][pilih], d["y"][pilih], d["idx"][pilih]
        bag, murni = bagian[pilih], d["murni"][pilih]
        it = int(d["i_target"])
        info = dict(pos=[str(p) for p in d["pos"]], kanal=[str(k) for k in d["kanal"]],
                    rata=float(d["rata"][it]), std=float(d["std"][it]),
                    atribut_c=d["atribut_c"].copy(), geser_daymet=int(d["geser_daymet_hari"]))
    pagar_uji(idx, bag)
    l, v = bag == 0, bag == 1
    assert idx[l].max() < idx[v].min(), "urutan latih-validasi tidak kronologis"
    return dict(h=h, varian=varian, i_target=it,
                X_latih=X[l], y_latih=y[l], idx_latih=idx[l],
                X_val=X[v], y_val=y[v], idx_val=idx[v], murni_val=murni[v],
                tanggal_val=PERIODE[idx[v]], **info)


def muat_deret_target():
    """Debit pos target (m³/s) dari hari pertama sampai hari terakhir bagian validasi."""
    q = pd.read_csv(PROC / "debit_isi_m3s.csv", index_col=0, parse_dates=True, dtype={TARGET: float},
                    nrows=I_VAL)[TARGET]
    assert len(q) == I_VAL and q.index[0] == PERIODE[0] and q.index[-1] < PERIODE[I_VAL], \
        "PAGAR: deret target memuat bagian uji"
    return q


def muat_uji(h, varian="V17"):
    """Bagian uji (kode 2 dan 3). Hanya untuk notebook 03, setelah konfigurasi final dikunci."""
    assert KONFIG_FINAL.exists(), "konfigurasi_final.json belum ada: bagian uji belum boleh dibuka"
    konf = json.loads(KONFIG_FINAL.read_text())
    assert konf.get("terkunci") is True, "konfigurasi_final.json belum dikunci"
    with np.load(VARIAN_FOLDER[varian] / f"tensor_h{h}.npz", allow_pickle=True) as d:
        u = d["bagian"] >= 2
        it = int(d["i_target"])
        return dict(h=h, varian=varian, i_target=it, X_uji=d["X"][u], y_uji=d["y"][u], idx_uji=d["idx"][u],
                    bagian_uji=d["bagian"][u], rata=float(d["rata"][it]), std=float(d["std"][it]))


def ke_m3s(z, data):
    """Kembalikan target dari skala z-score ke m³/s."""
    return np.asarray(z, float) * data["std"] + data["rata"]


# ---------------------------------------------------------------------------
# Metrik (dihitung dalam m³/s)
# ---------------------------------------------------------------------------
def nse(o, s):
    o, s = np.asarray(o, float), np.asarray(s, float)
    return 1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2)

def kge(o, s):
    o, s = np.asarray(o, float), np.asarray(s, float)
    r = np.corrcoef(o, s)[0, 1]
    return 1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2)

def rmse(o, s):
    return float(np.sqrt(np.mean((np.asarray(s, float) - np.asarray(o, float)) ** 2)))

def mae(o, s):
    return float(np.mean(np.abs(np.asarray(s, float) - np.asarray(o, float))))

def metrik(o, s):
    return {"NSE": float(nse(o, s)), "KGE": float(kge(o, s)), "RMSE": rmse(o, s), "MAE": mae(o, s)}


# ---------------------------------------------------------------------------
# Perangkat
# ---------------------------------------------------------------------------
_MESIN = None

def siapkan_tf():
    """Memori GPU dialokasikan bertahap (supaya beberapa proses bisa berbagi satu GPU) dan
    operasi deterministik diaktifkan. Aman dipanggil berulang kali."""
    import tensorflow as tf
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:          # GPU sudah diinisialisasi; pengaturan sebelumnya tetap berlaku
            pass
    tf.config.experimental.enable_op_determinism()


def mesin():
    """Penanda perangkat keras, misalnya 'arm64-CPU' atau 'x86_64-NVIDIA GeForce RTX 4060'.
    Hasil numerik berbeda antarperangkat, jadi run hanya dipakai ulang pada perangkat yang sama."""
    global _MESIN
    if _MESIN is None:
        import tensorflow as tf
        gpu = tf.config.list_physical_devices("GPU")
        nama = "+".join(tf.config.experimental.get_device_details(g).get("device_name", "GPU") for g in gpu)
        _MESIN = f"{platform.machine()}-{nama or 'CPU'}"
    return _MESIN


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
TETAP = {  # pengaturan yang tidak dituning (brief bagian 1)
    "optimizer": "adam", "batch": 256, "loss": "mse", "epoch_maks": 300,
    "patience": 50, "min_delta": 1e-5, "inisialisasi": "he_normal",
}
TENGAH = {  # pengaturan tengah tahap A
    "lr": 1e-3, "unit": 64, "lapisan_lstm": 2, "dropout": 0.25, "kernel": 2, "lapisan_konv": 1,
}


def bentuk_masukan(X, model, i_target):
    """Susun tensor (sampel, jendela, pos, kanal) sesuai masukan tiap model."""
    if model == "cnn_lstm":
        return X
    if model == "s_lstm":                       # 17 pos × 10 kanal diratakan, tanpa sumbu pos
        return X.reshape(X.shape[0], X.shape[1], -1)
    if model == "lstm_target":                  # kanal pos target saja
        return X[:, :, i_target, :]
    raise ValueError(model)


def bangun_model(model, cfg, bentuk):
    """Bangun model Keras. bentuk = bentuk satu sampel masukan (tanpa sumbu sampel)."""
    siapkan_tf()
    import keras
    from keras import layers
    init = TETAP["inisialisasi"]
    x_in = keras.Input(shape=bentuk)
    x = x_in
    if model == "cnn_lstm":
        # Conv1D di sumbu pos pada setiap langkah waktu; ditulis sebagai Conv2D dengan kernel 1×k,
        # yang secara matematis sama dengan Conv1D yang dibagi bobotnya di semua langkah waktu
        for _ in range(cfg["lapisan_konv"]):
            x = layers.Conv2D(cfg["unit"], (1, cfg["kernel"]), padding="valid", activation="relu",
                              kernel_initializer=init)(x)
        x = layers.Dropout(cfg["dropout"])(x)
        x = layers.Reshape((bentuk[0], -1))(x)          # ratakan: (jendela, pos' × filter)
    for j in range(cfg["lapisan_lstm"]):
        akhir = j == cfg["lapisan_lstm"] - 1
        x = layers.LSTM(cfg["unit"], return_sequences=not akhir, kernel_initializer=init)(x)
        x = layers.Dropout(cfg["dropout"])(x)
    y = layers.Dense(1, kernel_initializer=init)(x)
    m = keras.Model(x_in, y, name=model)
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg["lr"]), loss=TETAP["loss"])
    return m


def latih(model, cfg, data, seed, verbose=0, epoch_maks=None):
    """Latih satu model pada satu horizon dan satu seed. Validasi dipakai untuk early stopping.

    Mengembalikan (model_keras, ringkasan), ringkasan berisi epoch terbaik, jumlah epoch,
    waktu latih, prediksi validasi (m³/s), dan metrik validasi."""
    import keras
    siapkan_tf()
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    Xl = bentuk_masukan(data["X_latih"], model, data["i_target"])
    Xv = bentuk_masukan(data["X_val"], model, data["i_target"])
    m = bangun_model(model, cfg, Xl.shape[1:])
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=TETAP["patience"],
                                       min_delta=TETAP["min_delta"], restore_best_weights=True)
    t0 = time.perf_counter()
    hist = m.fit(Xl, data["y_latih"], validation_data=(Xv, data["y_val"]),
                 epochs=epoch_maks or TETAP["epoch_maks"], batch_size=TETAP["batch"],
                 shuffle=True, callbacks=[es], verbose=verbose)
    detik = time.perf_counter() - t0
    vl = np.asarray(hist.history["val_loss"])
    pred = ke_m3s(m.predict(Xv, batch_size=2048, verbose=0).ravel(), data)
    obs = ke_m3s(data["y_val"], data)
    ringkas = dict(epoch_terbaik=int(vl.argmin()) + 1, n_epoch=len(vl), waktu_latih_s=round(detik, 2),
                   val_loss_terbaik=float(vl.min()), prediksi=pred, observasi=obs,
                   n_param=int(m.count_params()), riwayat=hist.history, **metrik(obs, pred))
    return m, ringkas


# ---------------------------------------------------------------------------
# Log run
# ---------------------------------------------------------------------------
KOLOM_LOG = ["waktu", "id_run", "mesin", "notebook", "tahap", "model", "varian", "jendela", "hujan", "transformasi",
             "horizon", "seed", "konfigurasi", "n_param", "epoch_terbaik", "n_epoch", "waktu_latih_s",
             "val_loss_terbaik", "NSE", "KGE", "RMSE", "MAE"]


def catat_run(notebook, tahap, model, cfg, data, seed, ringkas, jendela=14, hujan="daymet",
              transformasi="log1p"):
    """Tambahkan satu baris ke log_run.csv dan simpan prediksi validasinya."""
    id_run = (f"{tahap}_{model}_{data['varian']}_w{jendela}_{hujan}_{transformasi}_h{data['h']}_s{seed}_"
              f"{datetime.now():%Y%m%d%H%M%S%f}")
    baris = dict(waktu=datetime.now().isoformat(timespec="seconds"), id_run=id_run, mesin=mesin(), notebook=notebook,
                 tahap=tahap, model=model, varian=data["varian"], jendela=jendela, hujan=hujan,
                 transformasi=transformasi, horizon=data["h"], seed=seed,
                 konfigurasi=json.dumps({**cfg, **TETAP}, sort_keys=True),
                 **{k: ringkas[k] for k in ["n_param", "epoch_terbaik", "n_epoch", "waktu_latih_s",
                                            "val_loss_terbaik", "NSE", "KGE", "RMSE", "MAE"]})
    pd.DataFrame([baris], columns=KOLOM_LOG).to_csv(LOG_RUN, mode="a", header=not LOG_RUN.exists(), index=False)
    np.savez_compressed(PREDIKSI / f"{id_run}.npz", idx_val=data["idx_val"], prediksi=ringkas["prediksi"],
                        observasi=ringkas["observasi"], loss=np.asarray(ringkas["riwayat"]["loss"]),
                        val_loss=np.asarray(ringkas["riwayat"]["val_loss"]))
    return id_run


def baca_log():
    return pd.read_csv(LOG_RUN, dtype={"varian": str}) if LOG_RUN.exists() else pd.DataFrame(columns=KOLOM_LOG)


def jalankan(notebook, tahap, model, cfg, data, seed, jendela=14, hujan="daymet", transformasi="log1p",
             ulang=False, verbose=0):
    """Latih dan catat satu run, atau muat hasilnya bila run yang sama persis sudah ada di log.

    Run dianggap sama bila perangkat, model, varian, jendela, sumber hujan, transformasi, horizon, seed,
    dan konfigurasi lengkapnya sama, apa pun notebook atau tahap yang pertama menjalankannya."""
    konf = json.dumps({**cfg, **TETAP}, sort_keys=True)
    log = baca_log()
    if not ulang and len(log):
        cocok = log[(log["mesin"] == mesin()) & (log["model"] == model) & (log["varian"] == data["varian"])
                    & (log["jendela"] == jendela)
                    & (log["hujan"] == hujan) & (log["transformasi"] == transformasi)
                    & (log["horizon"] == data["h"]) & (log["seed"] == seed) & (log["konfigurasi"] == konf)]
        if len(cocok):
            b = cocok.iloc[-1]
            with np.load(PREDIKSI / f"{b['id_run']}.npz") as f:
                assert np.array_equal(f["idx_val"], data["idx_val"]), "indeks validasi berbeda dengan run tercatat"
                hasil = dict(b) | dict(prediksi=f["prediksi"], observasi=f["observasi"],
                                       riwayat={"loss": f["loss"], "val_loss": f["val_loss"]}, dari_log=True)
            return hasil
    _, r = latih(model, cfg, data, seed, verbose=verbose)
    id_run = catat_run(notebook, tahap, model, cfg, data, seed, r, jendela, hujan, transformasi)
    return r | dict(id_run=id_run, model=model, horizon=data["h"], seed=seed, dari_log=False)
