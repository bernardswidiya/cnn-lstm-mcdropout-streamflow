#!/usr/bin/env python3
"""
Unduh hujan tambahan untuk preprocessing v2: stasiun GHCN-Daily di sekitar DAS 02035000 dan
COOP Hourly Precipitation Data (HPD) versi 2 untuk tiga stasiun.

File lama di raw/ tidak diubah. Sebelum mengunduh, sidik jari SHA-256 semua file yang sudah ada
di raw/ dicatat sekali ke raw/hujan/hash_raw_lama.sha256 (tidak ditimpa bila sudah ada), supaya
notebook bisa memeriksa bahwa file lama utuh. File .dly yang sudah ada tidak diunduh ulang.

Keluaran (semuanya file baru):
  raw/hujan/hash_raw_lama.sha256       SHA-256 file raw/ sebelum unduhan v2
  raw/hujan/stasiun_v2.txt             baris ghcnd-stations.txt untuk semua kandidat (format asli)
  raw/hujan/inventory_v2.txt           baris PRCP ghcnd-inventory.txt untuk kandidat (format asli)
  raw/hujan/dly/<ID>.dly               .dly kandidat yang belum ada
  raw/hujan/hpd/<ID>.csv               HPD v2 per stasiun (format asli)
  raw/hujan/hpd/HPD_v02r02_stationinv_*.csv   inventaris stasiun HPD
  raw/hujan/hpd/readme.csv.txt         dokumentasi format HPD

Kandidat: stasiun USC dan USW di ghcnd-stations.txt yang berjarak paling jauh 50 km dari batas
poligon DAS 02035000 (raw/network/basin/02035000.json, jarak dihitung di EPSG:5070; stasiun di dalam
DAS berjarak 0), dengan rekaman PRCP di ghcnd-inventory.txt yang mulai paling lambat 2001 dan
berakhir paling awal 2024. Kelengkapan sebenarnya dihitung ulang dari .dly di notebook.
ghcnd-stations.txt dan ghcnd-inventory.txt lengkap (11 MB dan 35 MB) hanya disimpan sementara.
"""
import argparse
import hashlib
import sys
import tempfile
import time
from pathlib import Path

import geopandas as gpd
import requests
from shapely.geometry import Point

GHCN = "https://www.ncei.noaa.gov/pub/data/ghcn/daily"
HPD = "https://www.ncei.noaa.gov/data/coop-hourly-precipitation/v2"
STASIUN_HPD = ["USC00443310", "USC00446712", "USC00445595"]     # Gathright Dam, Piedmont, Mill Gap


def ambil(url, coba=4):
    for i in range(coba):
        try:
            r = requests.get(url, timeout=600)
            r.raise_for_status()
            return r.content
        except Exception as e:
            if i == coba - 1:
                raise
            print(f"      ulang ({e.__class__.__name__}) ...")
            time.sleep(5 * (i + 1))


def tulis_baru(path, isi):
    """Tulis file baru; file yang sudah ada tidak ditimpa."""
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(isi)
    return True


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blok in iter(lambda: f.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw")
    ap.add_argument("--radius", type=float, default=50.0, help="jarak maksimum ke batas DAS, km")
    ap.add_argument("--muara", default="02035000")
    a = ap.parse_args()
    raw = Path(a.raw)
    hujan = raw / "hujan"

    print("[0/4] sidik jari file raw/ yang sudah ada ...")
    manifest = hujan / "hash_raw_lama.sha256"
    if manifest.exists():
        print(f"      {manifest} sudah ada, tidak ditimpa")
    else:
        files = sorted(p for p in raw.rglob("*") if p.is_file())
        teks = "".join(f"{sha256(p)}  {p.relative_to(raw).as_posix()}\n" for p in files)
        tulis_baru(manifest, teks.encode())
        print(f"      {len(files)} file -> {manifest}")

    print("[1/4] daftar stasiun dan inventaris GHCN-Daily ...")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "st.txt").write_bytes(ambil(f"{GHCN}/ghcnd-stations.txt"))
        (tmp / "inv.txt").write_bytes(ambil(f"{GHCN}/ghcnd-inventory.txt"))
        baris_st = (tmp / "st.txt").read_text().splitlines()
        baris_inv = (tmp / "inv.txt").read_text().splitlines()

    das = gpd.read_file(raw / "network" / "basin" / f"{a.muara}.json").to_crs(5070).geometry.union_all()
    kotak = das.buffer(a.radius * 1000 + 5000)
    us = [b for b in baris_st if b[:3] in ("USC", "USW")]
    titik = gpd.GeoSeries([Point(float(b[21:30]), float(b[12:20])) for b in us], crs=4326).to_crs(5070)
    jarak = titik.distance(das) / 1000.0                      # 0 di dalam DAS
    dekat = {b[:11]: (b, float(jarak.iat[i])) for i, b in enumerate(us)
             if kotak.contains(titik.iat[i]) and jarak.iat[i] <= a.radius}
    inv = {b[:11]: b for b in baris_inv if b[31:35] == "PRCP" and b[:11] in dekat}
    kandidat = sorted(s for s, b in inv.items() if int(b[36:40]) <= 2001 and int(b[41:45]) >= 2024)
    print(f"      USC/USW dalam {a.radius:.0f} km dari batas DAS: {len(dekat)}; "
          f"PRCP mulai <= 2001 dan berakhir >= 2024: {len(kandidat)}")
    tulis_baru(hujan / "stasiun_v2.txt", ("\n".join(dekat[s][0] for s in kandidat) + "\n").encode())
    tulis_baru(hujan / "inventory_v2.txt", ("\n".join(inv[s] for s in kandidat) + "\n").encode())

    print(f"[2/4] .dly kandidat yang belum ada ...")
    baru = 0
    for i, s in enumerate(kandidat, 1):
        f = hujan / "dly" / f"{s}.dly"
        if f.exists():
            print(f"      [{i}/{len(kandidat)}] {s}: sudah ada, dilewati")
            continue
        tulis_baru(f, ambil(f"{GHCN}/all/{s}.dly"))
        baru += 1
        print(f"      [{i}/{len(kandidat)}] {s} ({dekat[s][1]:.1f} km): {f.stat().st_size / 1e6:.2f} MB")
        time.sleep(0.3)

    print("[3/4] HPD v2: inventaris dan dokumentasi ...")
    daftar = ambil(f"{HPD}/station-inventory/").decode()
    nama_inv = sorted(set(t.split('"')[0] for t in daftar.split('href="')[1:] if t.startswith("HPD_v02r02_stationinv")))[-1]
    tulis_baru(hujan / "hpd" / nama_inv, ambil(f"{HPD}/station-inventory/{nama_inv}"))
    tulis_baru(hujan / "hpd" / "readme.csv.txt", ambil(f"{HPD}/doc/readme.csv.txt"))

    print(f"[4/4] HPD v2: {len(STASIUN_HPD)} stasiun ...")
    for s in STASIUN_HPD:
        f = hujan / "hpd" / f"{s}.csv"
        ok = tulis_baru(f, ambil(f"{HPD}/access/{s}.csv"))
        print(f"      {s}: {f.stat().st_size / 1e6:.2f} MB{'' if ok else ' (sudah ada)'}")

    print(f"\nSelesai. {len(kandidat)} kandidat, {baru} .dly baru, {len(STASIUN_HPD)} stasiun HPD.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
