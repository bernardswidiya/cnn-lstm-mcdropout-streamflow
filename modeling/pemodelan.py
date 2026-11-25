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
# Proses pekerja menulis ke log sendiri (log_run_p0.csv, log_run_p1.csv, ...) yang digabung oleh gabung_log
LOG_PROSES = HASIL / os.environ["PM_LOG_RUN"] if os.environ.get("PM_LOG_RUN") else LOG_RUN
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


def muat_latih_val(h, varian="V17", jendela=14, hujan="daymet", transformasi="log1p"):
    """Tensor horizon h untuk bagian latih dan validasi saja.

    Pilihan dasar (jendela 14, hujan Daymet tergeser, log1p) dibaca langsung dari processed/. Pilihan lain
    tahap A disusun ulang di memori dengan `tensor_pilihan` (fungsi susun preprocessing tahap 24).

    Mengembalikan dict berisi X_latih, y_latih, idx_latih, X_val, y_val, idx_val,
    i_target, pos, kanal, skala debit pos target (rata, std), dan tanggal target.
    Bagian uji dibuang di dalam fungsi ini, sebelum apa pun dikembalikan."""
    if varian == "V17_daymet_asli":
        raise ValueError("V17_daymet_asli tidak ikut seleksi (kebocoran ±7 jam); hanya untuk notebook 03")
    if hujan.startswith("v2_"):
        return muat_latih_val_v2(h, hujan[3:], varian, jendela, transformasi)
    with np.load(VARIAN_FOLDER[varian] / f"tensor_h{h}.npz", allow_pickle=True) as d:
        it = int(d["i_target"])
        info = dict(pos=[str(p) for p in d["pos"]], kanal=[str(k) for k in d["kanal"]],
                    rata=float(d["rata"][it]), std=float(d["std"][it]),
                    atribut_c=d["atribut_c"].copy(), geser_daymet=int(d["geser_daymet_hari"]))
        if (jendela, hujan, transformasi) == PILIHAN_DASAR:
            bagian = d["bagian"]
            pilih = bagian <= 1
            X, y, idx = d["X"][pilih], d["y"][pilih], d["idx"][pilih]
            bag, murni = bagian[pilih], d["murni"][pilih]
    if (jendela, hujan, transformasi) != PILIHAN_DASAR:
        X, y, idx, bag, murni = tensor_pilihan(h, varian, jendela, hujan, transformasi)
        pilih = bag <= 1
        X, y, idx, bag, murni = X[pilih], y[pilih], idx[pilih], bag[pilih], murni[pilih]
    pagar_uji(idx, bag)
    l, v = bag == 0, bag == 1
    assert idx[l].max() < idx[v].min(), "urutan latih-validasi tidak kronologis"
    return dict(h=h, varian=varian, jendela=jendela, hujan=hujan, transformasi=transformasi, i_target=it,
                X_latih=X[l], y_latih=y[l], idx_latih=idx[l],
                X_val=X[v], y_val=y[v], idx_val=idx[v], murni_val=murni[v],
                tanggal_val=PERIODE[idx[v]], **info)


V2 = PROC / "v2"


def varian_v2():
    """Daftar varian hujan v2 (preprocessing Bagian VIII): kode -> kanal hujan dan jumlah kanal per pos."""
    return json.loads((V2 / "varian_v2.json").read_text())["varian"]


# ablasi kontribusi data: kanal yang dipertahankan (kelompok kumulatif); "hujan" = semua kanal hujan varian
ABLASI_V2 = {"debit": ["debit_terskala", "flag_isian"],
             "debit_hujan": ["debit_terskala", "flag_isian", "hujan"],
             "debit_hujan_suhu": ["debit_terskala", "flag_isian", "hujan", "tmax_terskala", "tmin_terskala"]}
ATR_A = ["log_luas", "elev_rata_das_m", "kemiringan_das_m_per_km", "hujan_tahunan_mm"]   # tingkat A, tahap 13


def muat_latih_val_v2(h, kode, varian="V17", jendela=8, transformasi="log1p"):
    """Tensor v2 (processed/v2/tensor_h{h}.npz) untuk satu varian hujan R0-R5, bagian latih dan validasi saja.
    X = debit, flag, kanal hujan varian, Tmax, Tmin, 5 atribut. Bagian uji dibuang sebelum dikembalikan.
    kode boleh diberi akhiran '@<ablasi>' (kunci ABLASI_V2) untuk hanya memakai sebagian kanal."""
    kode, _, ablasi = kode.partition("@")
    info_v = varian_v2()[kode]
    with np.load(V2 / f"tensor_h{h}.npz", allow_pickle=True) as d:
        assert (varian, jendela, transformasi) == (str(d["varian"]), int(d["jendela"]), "log1p"), \
            "tensor v2 hanya tersedia untuk V17, jendela 8 hari, log1p"
        it, pos_h = int(d["i_target"]), int(d["posisi_hujan"])
        bagian = d["bagian"]
        pilih = bagian <= 1
        nama_h = [str(k) for k in d["kanal_hujan"]]
        j = [nama_h.index(k) for k in info_v["kanal_hujan"]]
        Xd, Xh = d["X_dasar"][pilih], d["X_hujan"][pilih][..., j]
        y, idx, bag, murni = d["y"][pilih], d["idx"][pilih], bagian[pilih], d["murni"][pilih]
        kanal_d = [str(k) for k in d["kanal_dasar"]]
        info = dict(pos=[str(p) for p in d["pos"]], rata=float(d["rata"][it]), std=float(d["std"][it]),
                    atribut_c=d["atribut_c"].copy(), geser_daymet=1)
    pagar_uji(idx, bag)
    X = np.concatenate([Xd[..., :pos_h], Xh, Xd[..., pos_h:]], axis=-1)
    assert X.shape[-1] == info_v["kanal_per_pos"]
    kanal = kanal_d[:pos_h] + info_v["kanal_hujan"] + kanal_d[pos_h:]
    if ablasi == "atribut_acak":
        # kontrol [Li2022], sama dengan varian_ablasi preprocessing tahap 30: empat atribut tingkat A diganti
        # vektor normal baku tetap per pos (benih 0); bfi_latih (tingkat B) dan kanal dinamis tidak berubah
        j_a = [kanal.index(k) for k in ATR_A]
        acak = np.random.default_rng(0).standard_normal((X.shape[2], len(ATR_A))).astype(X.dtype)
        X = X.copy()
        X[..., j_a] = acak[None, None, :, :]
    elif ablasi:
        simpan = [k for g in ABLASI_V2[ablasi] for k in (info_v["kanal_hujan"] if g == "hujan" else [g])]
        j_ = [kanal.index(k) for k in simpan]
        assert j_[0] == 0, "kanal debit harus tetap kanal 0"
        X, kanal = X[..., j_], simpan
    l, v = bag == 0, bag == 1
    assert idx[l].max() < idx[v].min(), "urutan latih-validasi tidak kronologis"
    return dict(h=h, varian=varian, jendela=jendela, hujan=f"v2_{kode}" + (f"@{ablasi}" if ablasi else ""),
                transformasi=transformasi, i_target=it,
                X_latih=X[l], y_latih=y[l], idx_latih=idx[l], X_val=X[v], y_val=y[v], idx_val=idx[v],
                murni_val=murni[v], tanggal_val=PERIODE[idx[v]], kanal=kanal, **info)


# ---------------------------------------------------------------------------
# Tensor untuk pilihan data tahap A (jendela, sumber hujan, transformasi hujan)
# ---------------------------------------------------------------------------
PILIHAN_DASAR = (14, "daymet", "log1p")
K_HUJAN = 2                                     # indeks kanal hujan di tensor
HUJAN = {"daymet": "Daymet tergeser", "ghcn_idw": "GHCN-Daily, pemetaan IDW",
         "ghcn_idw30": "GHCN-Daily, IDW radius 30 km", "ghcn_idw80": "GHCN-Daily, IDW radius 80 km",
         "ghcn_terdekat": "GHCN-Daily, stasiun terdekat",
         "ghcn_idw_tanpa_bandara": "GHCN-Daily, IDW tanpa 3 stasiun bandara"}
BANDARA = ["USW00013741", "USW00013733", "USW00093736"]     # Roanoke, Lynchburg, Charlottesville
# pemetaan GHCN: metode, radius (km), stasiun yang dikeluarkan
PEMETAAN_GHCN = {"ghcn_idw": ("idw", 50.0, ()), "ghcn_idw30": ("idw", 30.0, ()), "ghcn_idw80": ("idw", 80.0, ()),
                 "ghcn_terdekat": ("terdekat", None, ()), "ghcn_idw_tanpa_bandara": ("idw", 50.0, tuple(BANDARA))}
TRANSFORMASI = {"log1p": "ln(1 + x)", "log10_akar": "log10(√x + 0,1)"}


def susun(arr, i_target, window, horizon, sah, tak_murni):
    """Salinan persis fungsi susun dari preprocessing tahap 24."""
    X, y, idx, murni = [], [], [], []
    for t in range(window-1, len(arr)-horizon):
        a, b = t-window+1, t+1
        if not sah[a:b].all() or not sah[t+horizon]:
            continue
        X.append(arr[a:b])
        y.append(arr[t+horizon, i_target, 0])
        idx.append(t+horizon)
        # True kalau target hasil ukur, bukan perkiraan
        murni.append(not tak_murni[t+horizon, i_target])
    return np.asarray(X), np.asarray(y), np.asarray(idx), np.asarray(murni)


def _harian_dari_jendela(X, idx, h):
    """Deret harian (hari, ...) dari tensor berjendela. Setiap hari input ditulis dari semua sampel yang
    memuatnya, lalu diperiksa bahwa semua salinan hari yang sama identik."""
    w = X.shape[1]
    hari = (idx - h)[:, None] + np.arange(-(w - 1), 1)[None, :]
    out = np.full((N_HARI,) + X.shape[2:], np.nan, dtype=X.dtype)
    out[hari.ravel()] = X.reshape((-1,) + X.shape[2:])
    assert np.array_equal(out[hari], X), "salinan hari yang sama berbeda antarsampel"
    return out


def kode_e_target():
    """True pada hari ketika debit pos target berkode perkiraan (e), dari file NWIS mentah (hanya dibaca).
    Hanya sampai hari terakhir bagian validasi."""
    rdb = pd.read_csv(ROOT / "raw" / "discharge" / f"dv_{TARGET}.rdb", sep="\t", comment="#", dtype=str).iloc[1:]
    kode = rdb.set_index(pd.to_datetime(rdb["datetime"])).iloc[:, -1]
    e = kode.reindex(PERIODE[:I_VAL]).fillna("").str.contains("e").to_numpy()
    return np.r_[e, np.zeros(N_HARI - I_VAL, bool)]


def _jendela_latih_val(berkas, nama_X="X"):
    """X, y, idx sampel latih dan validasi saja dari satu file tensor; sampel uji dibuang sebelum dipakai."""
    with np.load(berkas, allow_pickle=True) as d:
        pilih = d["bagian"] <= 1
        out = d[nama_X][pilih], (d["y"][pilih] if "y" in d.files else None), d["idx"][pilih], d["bagian"][pilih]
    pagar_uji(out[2], out[3])
    return out[:3]


_HARIAN = {}

def deret_harian(varian="V17"):
    """Rekonstruksi larik harian (hari, pos, 10 kanal) dari sampel latih dan validasi tensor di processed/.

    Hanya hari 0 sampai I_VAL-1 yang terisi; hari bagian uji tetap NaN, sehingga susun tidak dapat membentuk
    sampel uji. Sampel latih dan validasi h1 memuat setiap hari 0..I_VAL-2 sebagai input. Hari I_VAL-1 hanya
    pernah menjadi target, jadi hanya debit pos targetnya yang diisi (dari y sampel validasi terakhir);
    kanal lain hari itu diisi 0 supaya lolos pemeriksaan `sah`. Nilai itu tidak pernah masuk ke X, karena
    input paling akhir sampel validasi adalah hari I_VAL-1-h."""
    if varian in _HARIAN:
        return _HARIAN[varian]
    berkas = VARIAN_FOLDER[varian] / "tensor_h1.npz"
    with np.load(berkas, allow_pickle=True) as d:
        it = int(d["i_target"])
        info = {k: d[k].copy() for k in ["rata_hujan", "std_hujan"]}
        info["pos"] = [str(p) for p in d["pos"]]
    X, y, idx = _jendela_latih_val(berkas)
    arr = _harian_dari_jendela(X, idx, 1)
    ada = np.isfinite(arr).all(axis=(1, 2))
    assert idx[-1] == I_VAL - 1 and ada[:I_VAL - 1].all() and not ada[I_VAL - 1:].any()
    arr[I_VAL - 1] = 0.0
    arr[I_VAL - 1, it, 0] = y[-1]
    for h in (2, 3):                           # tensor h2 dan h3 harus memuat hari yang sama dengan nilai yang sama
        X_h, _, idx_h = _jendela_latih_val(VARIAN_FOLDER[varian] / f"tensor_h{h}.npz")
        a_h = _harian_dari_jendela(X_h, idx_h, h)
        ada_h = np.isfinite(a_h).all(axis=(1, 2))
        assert np.array_equal(a_h[ada_h], arr[ada_h]), f"tensor h{h} tidak sejajar dengan h1"
    tak = np.zeros(arr.shape[:2], bool)
    tak[:, it] = kode_e_target()
    _HARIAN[varian] = dict(arr=arr, i_target=it, tak=tak, **info)
    return _HARIAN[varian]


def hujan_ghcn_harian():
    """Hujan GHCN-Daily IDW (log1p terskala) per hari dari sampel latih dan validasi tensor_hujan_h1.npz;
    hanya ada untuk pos V17."""
    if "ghcn" in _HARIAN:
        return _HARIAN["ghcn"]
    berkas = PROC / "tensor_hujan_h1.npz"
    X, _, idx = _jendela_latih_val(berkas, "X_idw")
    with np.load(berkas, allow_pickle=True) as d:
        _HARIAN["ghcn"] = dict(z=_harian_dari_jendela(X, idx, 1)[..., 0], rata=d["rata_idw"].copy(),
                               std=d["std_idw"].copy(), pos=[str(p) for p in d["pos"]])
    return _HARIAN["ghcn"]


# --- hujan GHCN IDW untuk pos di luar V17: fungsi preprocessing tahap 17b disalin apa adanya
R_IDW = 50.0


def jarak_km(a1, o1, a2, o2):
    """Jarak haversine (km) antara dua titik (lintang, bujur) dalam derajat. Salinan preprocessing."""
    a1, o1, a2, o2 = map(np.radians, (a1, o1, a2, o2))
    h = np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin((o2 - o1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(h))


def petakan_terdekat(H, D):
    """Salinan preprocessing tahap 17b."""
    out, cad = {}, {}
    for p in D.index:
        urut = D.loc[p].sort_values().index
        x = H[urut[0]].copy()
        c = pd.Series(False, index=H.index)
        for s in urut[1:]:
            k = x.isna() & H[s].notna()
            x[k] = H.loc[k, s]; c |= k
        out[p], cad[p] = x, c
    return pd.DataFrame(out), pd.DataFrame(cad)


def petakan_idw(H, D, radius):
    """Salinan preprocessing tahap 17b."""
    terd, _ = petakan_terdekat(H, D)
    out, cad = {}, {}
    for p in D.index:
        sel = D.loc[p][D.loc[p] <= radius]
        w = 1.0 / sel ** 2
        X = H[sel.index]
        bobot = X.notna().mul(w, axis=1)
        x = (X.fillna(0).mul(w, axis=1).sum(axis=1) / bobot.sum(axis=1)).where(bobot.sum(axis=1) > 0)
        c = x.isna()
        out[p], cad[p] = x.fillna(terd[p]), c
    return pd.DataFrame(out), pd.DataFrame(cad)


def ghcn_idw_mm(pos, metode="idw", radius=R_IDW, kecuali=()):
    """Hujan GHCN-Daily (mm/hari) untuk daftar pos, hari 0 sampai I_VAL-1, dengan petakan_idw atau
    petakan_terdekat preprocessing. Input stasiun adalah processed/hujan/hujan_harian_mm.csv (nilai .dly
    setelah pemeriksaan akumulasi dan penambalan tahap 16), koordinat pos dari raw/sites/site_info.rdb (hanya
    dibaca). `kecuali`: stasiun yang dikeluarkan sebelum pemetaan."""
    site = pd.read_csv(ROOT / "raw" / "sites" / "site_info.rdb", sep="\t", comment="#", dtype=str).iloc[1:]
    site = site.set_index("site_no")
    st = pd.read_csv(PROC / "hujan" / "stasiun_hujan.csv", index_col=0)
    H = pd.read_csv(PROC / "hujan" / "hujan_harian_mm.csv", index_col=0, parse_dates=True, nrows=I_VAL)
    assert list(H.columns) == list(st.index) and H.index[-1] == PERIODE[I_VAL - 1]
    st, H = st.drop(index=list(kecuali)), H.drop(columns=list(kecuali))
    la, lo = site.loc[pos, "dec_lat_va"].astype(float).values, site.loc[pos, "dec_long_va"].astype(float).values
    D = pd.DataFrame(jarak_km(la[:, None], lo[:, None], st["lintang"].values[None, :], st["bujur"].values[None, :]),
                     index=pos, columns=list(st.index))
    P, cad = petakan_idw(H, D, radius) if metode == "idw" else petakan_terdekat(H, D)
    return P[pos], cad[pos], D


def balik_log1p(z, rata, std):
    """Kembalikan hujan log1p terskala ke mm/hari. Sisa pembulatan float32 di sekitar nol dijadikan 0."""
    x = np.expm1(np.asarray(z, np.float64) * std + rata)
    return np.where(x < 1e-6, 0.0, x)


def skala_hujan(P, transformasi):
    """Transformasi lalu standardisasi per pos dengan rata-rata dan simpangan baku bagian latih (ddof=0),
    sama dengan skala_latih di preprocessing tahap 24."""
    L = np.log1p(P) if transformasi == "log1p" else np.log10(np.sqrt(P) + 0.1)
    r, s = np.nanmean(L[:I_LATIH], axis=0), np.nanstd(L[:I_LATIH], axis=0)
    s = np.where(s == 0, 1.0, s)
    return (L - r) / s, r, s


def kanal_hujan(varian, hujan, transformasi):
    """Kanal hujan harian (hari, pos) untuk satu pilihan. Pilihan log1p memakai nilai tersimpan apa adanya;
    log10(√x + 0,1) dihitung dari hujan mm yang dikembalikan dari nilai log1p."""
    H = deret_harian(varian)
    if hujan == "daymet":
        z, r, s = H["arr"][..., K_HUJAN].astype(np.float64), H["rata_hujan"], H["std_hujan"]
    elif hujan == "ghcn_idw" and varian == "V17":
        G = hujan_ghcn_harian()
        assert G["pos"] == H["pos"], "urutan pos tensor hujan GHCN berbeda dengan V17"
        z, r, s = G["z"].astype(np.float64), G["rata"], G["std"]
    elif hujan in PEMETAAN_GHCN:
        # pemetaan lain atau pos di luar V17: dipetakan ulang dengan fungsi preprocessing, lalu ditransformasi
        P = np.full((N_HARI, len(H["pos"])), np.nan)
        metode, radius, kecuali = PEMETAAN_GHCN[hujan]
        P[:I_VAL] = ghcn_idw_mm(H["pos"], metode, radius, kecuali)[0].to_numpy(float)
        P[I_VAL - 1:] = np.nan                              # hari I_VAL-1 tidak pernah menjadi input
        return skala_hujan(P, transformasi)[0]
    else:
        raise ValueError(hujan)
    if transformasi == "log1p":
        return z
    z_ = z.copy(); z_[I_VAL - 1:] = np.nan               # hari I_VAL-1 tidak pernah menjadi input
    P = balik_log1p(z_, r, s)
    return skala_hujan(P, transformasi)[0]


def tensor_pilihan(h, varian="V17", jendela=14, hujan="daymet", transformasi="log1p"):
    """Tensor horizon h untuk satu pilihan data, disusun dengan fungsi susun preprocessing dari deret harian
    yang hanya memuat hari latih dan validasi. Mengembalikan (X, y, idx, bagian, murni)."""
    H = deret_harian(varian)
    arr = H["arr"].copy()
    if (hujan, transformasi) != ("daymet", "log1p"):
        k = kanal_hujan(varian, hujan, transformasi)
        k[I_VAL - 1] = 0.0                               # sama dengan kanal lain hari itu (lihat deret_harian)
        arr[..., K_HUJAN] = k.astype(np.float32)
    sah = np.isfinite(arr).all(axis=(1, 2))
    X, y, idx, murni = susun(arr, H["i_target"], jendela, h, sah, H["tak"])
    bagian = np.where(idx < I_LATIH, 0, np.where(idx < I_VAL, 1, 2))
    bagian = np.where((bagian == 2) & (~murni), 3, bagian)
    pagar_uji(idx, bagian)
    return X.astype(np.float32), y.astype(np.float32), idx, bagian, murni


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


# Keluaran heteroskedastik (tahap C, [Sun2026] pers. 10, [Kendall2017]): Dense(2) = (mu, s = log sigma^2) dalam skala z
# target; s dijepit ke S_JEPIT di loss dan di prediksi supaya exp(-s) dan exp(s) tetap stabil.
S_JEPIT = (-10.0, 6.0)          # sigma_z antara e^-5 (0,007) dan e^3 (20) simpangan baku debit target


def heteroskedastik(model):
    return model.endswith("_het")


def nll_heteroskedastik(y_true, y_pred):
    """Loss per sampel exp(-s)(y - mu)^2 + s; rata-ratanya adalah pers. 10 Sun dkk. (2026) (tanpa konstanta)."""
    from keras import ops
    y = ops.reshape(ops.cast(y_true, y_pred.dtype), (-1,))
    mu, s = y_pred[:, 0], ops.clip(y_pred[:, 1], S_JEPIT[0], S_JEPIT[1])
    return ops.exp(-s) * ops.square(y - mu) + s


def bentuk_masukan(X, model, i_target):
    """Susun tensor (sampel, jendela, pos, kanal) sesuai masukan tiap model."""
    model = model.removesuffix("_het")
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
    het = heteroskedastik(model)
    model = model.removesuffix("_het")
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
    y = layers.Dense(2 if het else 1, kernel_initializer=init)(x)
    m = keras.Model(x_in, y, name=model + ("_het" if het else ""))
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg["lr"]),
              loss=nll_heteroskedastik if het else TETAP["loss"])
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
    keluar = m.predict(Xv, batch_size=2048, verbose=0)
    pred = ke_m3s(keluar[:, 0] if heteroskedastik(model) else keluar.ravel(), data)   # mu, dropout nonaktif
    obs = ke_m3s(data["y_val"], data)
    ringkas = dict(epoch_terbaik=int(vl.argmin()) + 1, n_epoch=len(vl), waktu_latih_s=round(detik, 2),
                   val_loss_terbaik=float(vl.min()), prediksi=pred, observasi=obs,
                   n_param=int(m.count_params()), riwayat=hist.history, **metrik(obs, pred))
    return m, ringkas


# ---------------------------------------------------------------------------
# Log run
# ---------------------------------------------------------------------------
def konfig_json(model, cfg):
    """Konfigurasi lengkap untuk log: cfg + pengaturan tetap; model heteroskedastik mencatat loss-nya sendiri."""
    k = {**cfg, **TETAP}
    if heteroskedastik(model):
        k["loss"] = "nll_heteroskedastik"
    return json.dumps(k, sort_keys=True)


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
                 konfigurasi=konfig_json(model, cfg),
                 **{k: ringkas[k] for k in ["n_param", "epoch_terbaik", "n_epoch", "waktu_latih_s",
                                            "val_loss_terbaik", "NSE", "KGE", "RMSE", "MAE"]})
    pd.DataFrame([baris], columns=KOLOM_LOG).to_csv(LOG_PROSES, mode="a", header=not LOG_PROSES.exists(), index=False)
    np.savez_compressed(PREDIKSI / f"{id_run}.npz", idx_val=data["idx_val"], prediksi=ringkas["prediksi"],
                        observasi=ringkas["observasi"], loss=np.asarray(ringkas["riwayat"]["loss"]),
                        val_loss=np.asarray(ringkas["riwayat"]["val_loss"]))
    return id_run


def _log_proses():
    return sorted(HASIL.glob("log_run_p*.csv"))


def baca_log():
    """Log utama ditambah log proses pekerja yang belum digabung. Proses pekerja hanya membaca log utama dan
    lognya sendiri (tugasnya tidak tumpang tindih dengan pekerja lain, dan log pekerja lain bisa sedang ditulis)."""
    daftar = [LOG_RUN, LOG_PROSES] if LOG_PROSES != LOG_RUN else [LOG_RUN, *_log_proses()]
    bagian = [pd.read_csv(f, dtype={"varian": str}) for f in daftar if f.exists()]
    return pd.concat(bagian, ignore_index=True) if bagian else pd.DataFrame(columns=KOLOM_LOG)


KUNCI_RUN = ["mesin", "model", "varian", "jendela", "hujan", "transformasi", "horizon", "seed", "konfigurasi"]


def gabung_log():
    """Gabungkan log proses pekerja ke log_run.csv. Gagal bila ada id_run ganda atau run yang sama
    (perangkat, data, konfigurasi, seed) tercatat lebih dari sekali."""
    files = _log_proses()
    if not files:
        return 0
    utama = pd.read_csv(LOG_RUN, dtype={"varian": str}) if LOG_RUN.exists() else pd.DataFrame(columns=KOLOM_LOG)
    baru = pd.concat([pd.read_csv(f, dtype={"varian": str}) for f in files], ignore_index=True)
    semua = pd.concat([utama, baru], ignore_index=True)
    assert not semua["id_run"].duplicated().any(), "id_run ganda saat menggabung log"
    assert not semua.duplicated(KUNCI_RUN).any(), "run yang sama tercatat lebih dari sekali"
    semua[KOLOM_LOG].to_csv(LOG_RUN, index=False)
    assert len(pd.read_csv(LOG_RUN)) == len(utama) + len(baru)
    for f in files:
        f.unlink()
    return len(baru)


def cari_run(model, cfg, varian, jendela, hujan, transformasi, h, seed, log=None):
    """Baris log untuk run yang sama persis di perangkat ini, atau None."""
    log = baca_log() if log is None else log
    if not len(log):
        return None
    konf = konfig_json(model, cfg)
    c = log[(log["mesin"] == mesin()) & (log["model"] == model) & (log["varian"] == varian)
            & (log["jendela"] == jendela) & (log["hujan"] == hujan) & (log["transformasi"] == transformasi)
            & (log["horizon"] == h) & (log["seed"] == seed) & (log["konfigurasi"] == konf)]
    return c.iloc[-1] if len(c) else None


def sudah_ada(model, cfg, varian, jendela, hujan, transformasi, h, seed, log=None):
    """True bila run yang sama persis sudah ada di log untuk perangkat ini."""
    return cari_run(model, cfg, varian, jendela, hujan, transformasi, h, seed, log) is not None


def _tugas_selesai(t, log):
    b = cari_run(t["model"], t["cfg"], t["varian"], t["jendela"], t["hujan"], t["transformasi"], t["h"], t["seed"], log)
    if b is None:
        return False
    return not t.get("finalis") or ((MC_VAL / f"{b['id_run']}.npz").exists() and (BOBOT / f"{b['id_run']}.weights.h5").exists())


def jalankan_paralel(tugas, n_proses=2):
    """Jalankan daftar tugas (dict: notebook, tahap, model, cfg, varian, jendela, hujan, transformasi, h, seed,
    dan opsional finalis=True untuk menyimpan bobot dan menghitung MC Dropout validasi)
    di n_proses proses pekerja yang berbagi satu GPU. Setiap proses menulis ke log_run_p{k}.csv; setelah semua
    selesai, log digabung dengan gabung_log. Tugas yang sudah selesai dilewati. Mengembalikan jumlah tugas yang
    dikerjakan."""
    import subprocess, sys
    log = baca_log()
    sisa = [t for t in tugas if not _tugas_selesai(t, log)]
    if not sisa:
        return 0
    n_baris_baru = sum(not sudah_ada(t["model"], t["cfg"], t["varian"], t["jendela"], t["hujan"], t["transformasi"],
                                     t["h"], t["seed"], log) for t in sisa)
    assert not _log_proses(), "masih ada log proses yang belum digabung"
    folder = ROOT / "cache" / "pekerja"
    folder.mkdir(parents=True, exist_ok=True)
    # urut menurut data supaya tiap proses memuat tensor yang sama berurutan; dibagi selang-seling
    sisa = sorted(sisa, key=lambda t: (t["varian"], t["jendela"], t["hujan"], t["transformasi"], t["h"], t["seed"]))
    proses = []
    for k in range(n_proses):
        bagian = sisa[k::n_proses]
        if not bagian:
            continue
        (folder / f"tugas_p{k}.json").write_text(json.dumps(bagian))
        env = {**os.environ, "PM_LOG_RUN": f"log_run_p{k}.csv", "TF_CPP_MIN_LOG_LEVEL": "3"}
        keluaran = open(folder / f"pekerja_p{k}.log", "w")
        proses.append((k, subprocess.Popen([sys.executable, str(MODELING / "pekerja.py"), str(folder / f"tugas_p{k}.json")],
                                           env=env, stdout=keluaran, stderr=subprocess.STDOUT), keluaran))
    gagal = []
    for k, pr, keluaran in proses:
        if pr.wait() != 0:
            gagal.append(k)
        keluaran.close()
    if gagal:
        raise RuntimeError(f"proses pekerja {gagal} gagal; lihat cache/pekerja/pekerja_p*.log")
    n = gabung_log()
    assert n == n_baris_baru, f"run baru di log {n}, seharusnya {n_baris_baru}"
    return len(sisa)


def jalankan(notebook, tahap, model, cfg, data, seed, jendela=None, hujan=None, transformasi=None,
             ulang=False, verbose=0):
    """Latih dan catat satu run, atau muat hasilnya bila run yang sama persis sudah ada di log.

    Run dianggap sama bila perangkat, model, varian, jendela, sumber hujan, transformasi, horizon, seed,
    dan konfigurasi lengkapnya sama, apa pun notebook atau tahap yang pertama menjalankannya.
    Jendela, sumber hujan, dan transformasi diambil dari data bila tidak diberikan."""
    jendela = data.get("jendela", 14) if jendela is None else jendela
    hujan = data.get("hujan", "daymet") if hujan is None else hujan
    transformasi = data.get("transformasi", "log1p") if transformasi is None else transformasi
    konf = konfig_json(model, cfg)
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


# ---------------------------------------------------------------------------
# Finalis tahap B: bobot tersimpan dan MC Dropout di bagian validasi
# ---------------------------------------------------------------------------
BOBOT = ROOT / "cache" / "bobot"          # bobot finalis (tidak masuk git)
MC_VAL = HASIL / "mc_val"                 # ringkasan MC Dropout validasi per run
N_LINTASAN = 1000                         # [Le2025]


def mc_dropout(m, X, lintasan, seed):
    """Prediksi dengan dropout aktif (training=True pada lapisan Dropout yang sama dengan saat latih)
    [Gal2016]; mengembalikan larik (lintasan, sampel) dalam skala z-score."""
    import keras, tensorflow as tf
    keras.utils.set_random_seed(seed)
    f = tf.function(lambda x: m(x, training=True))
    xv = tf.constant(X)
    out = np.empty((lintasan, len(X)), np.float32)
    for i in range(lintasan):
        out[i] = f(xv).numpy().ravel()
    return out


MC_HET = ROOT / "cache" / "mc_het"     # lintasan (mu, s) model heteroskedastik (tidak masuk git)
N_SAMPEL_HET = 10                         # sampel N(mu_i, sigma_i^2) per lintasan [Sun2026]


def benih_sampel_het(h, seed):
    return 2026_000 + 100 * h + seed


def lintasan_het(m, X, lintasan, seed):
    """MC Dropout untuk keluaran heteroskedastik: (mu, s) per lintasan, skala z, s dijepit ke S_JEPIT."""
    import keras, tensorflow as tf
    keras.utils.set_random_seed(seed)
    f = tf.function(lambda x: m(x, training=True))
    xv = tf.constant(X)
    mu, s = np.empty((lintasan, len(X)), np.float32), np.empty((lintasan, len(X)), np.float32)
    for i in range(lintasan):
        o = f(xv).numpy()
        mu[i], s[i] = o[:, 0], np.clip(o[:, 1], *S_JEPIT)
    return mu, s


def sampel_het(mu, s, data, benih, n=N_SAMPEL_HET):
    """n sampel N(mu_i, exp(s_i)) per lintasan, dikembalikan ke m3/s: larik (lintasan x n, sampel)."""
    rng = np.random.default_rng(benih)
    z = mu[:, None, :].astype(np.float64) + np.exp(s[:, None, :].astype(np.float64) / 2) * rng.standard_normal(
        (mu.shape[0], n, mu.shape[1]))
    return ke_m3s(z.reshape(-1, mu.shape[1]), data)


def jalankan_finalis(notebook, tahap, model, cfg, data, seed):
    """Run finalis: latih (atau latih ulang bila run sudah ada di log tetapi bobotnya belum disimpan), simpan
    bobot, lalu hitung MC Dropout 1.000 lintasan di bagian validasi. Latih ulang harus menghasilkan val_loss dan
    prediksi yang identik dengan run tercatat (determinisme per seed)."""
    BOBOT.mkdir(parents=True, exist_ok=True); MC_VAL.mkdir(parents=True, exist_ok=True)
    b = cari_run(model, cfg, data["varian"], data["jendela"], data["hujan"], data["transformasi"], data["h"], seed)
    if b is not None and (MC_VAL / f"{b['id_run']}.npz").exists() and (BOBOT / f"{b['id_run']}.weights.h5").exists():
        return jalankan(notebook, tahap, model, cfg, data, seed) | dict(latih_ulang=False)
    m, r = latih(model, cfg, data, seed)
    if b is not None:
        id_run = b["id_run"]
        with np.load(PREDIKSI / f"{id_run}.npz") as f:
            assert np.array_equal(f["val_loss"], np.asarray(r["riwayat"]["val_loss"])), f"latih ulang {id_run} tidak identik"
            assert np.array_equal(f["prediksi"], r["prediksi"]), f"prediksi latih ulang {id_run} tidak identik"
    else:
        id_run = catat_run(notebook, tahap, model, cfg, data, seed, r, data["jendela"], data["hujan"], data["transformasi"])
    m.save_weights(BOBOT / f"{id_run}.weights.h5")
    Xv = bentuk_masukan(data["X_val"], model, data["i_target"])
    t0 = time.perf_counter()
    tambahan = {}
    if heteroskedastik(model):
        # MC dua tahap [Sun2026]: lintasan (mu_i, s_i) disimpan di cache, lalu N_SAMPEL_HET sampel N(mu_i, e^s_i) per lintasan
        MC_HET.mkdir(parents=True, exist_ok=True)
        mu, s = lintasan_het(m, Xv, N_LINTASAN, seed)
        benih_s = benih_sampel_het(data["h"], seed)
        np.savez_compressed(MC_HET / f"{id_run}.npz", mu=mu, s=s, idx_val=data["idx_val"], benih_mc=seed,
                            benih_sampel=benih_s, n_sampel=N_SAMPEL_HET, s_jepit=np.array(S_JEPIT))
        S = sampel_het(mu, s, data, benih_s)
        tambahan = dict(prediksi_mc=S.mean(0), benih_mc=seed, benih_sampel=benih_s, n_sampel=N_SAMPEL_HET)
    else:
        S = ke_m3s(mc_dropout(m, Xv, N_LINTASAN, seed), data)
    q = np.quantile(S, [0.025, 0.16, 0.84, 0.975], axis=0)
    np.savez_compressed(MC_VAL / f"{id_run}.npz", idx_val=data["idx_val"], observasi=ke_m3s(data["y_val"], data),
                        prediksi=r["prediksi"], rata=S.mean(0), sb=S.std(0), q025=q[0], q16=q[1], q84=q[2], q975=q[3],
                        lintasan=N_LINTASAN, detik_mc=time.perf_counter() - t0, **tambahan)
    return r | dict(id_run=id_run, model=model, horizon=data["h"], seed=seed, dari_log=b is not None,
                    latih_ulang=b is not None)


def ringkas_mc(id_run):
    """PICP dan MPIW untuk ±1σ (nominal 68,3%) dan interval kuantil 2,5–97,5% (nominal 95%) dari satu run."""
    with np.load(MC_VAL / f"{id_run}.npz") as f:
        o, mu, sd, lo, hi = f["observasi"], f["rata"], f["sb"], f["q025"], f["q975"]
        return {"PICP68": float(np.mean((o >= mu - sd) & (o <= mu + sd))), "MPIW68": float(np.mean(2 * sd)),
                "PICP95": float(np.mean((o >= lo) & (o <= hi))), "MPIW95": float(np.mean(hi - lo)),
                "NSE_rata_mc": float(nse(o, mu)), "detik_mc": float(f["detik_mc"])}
