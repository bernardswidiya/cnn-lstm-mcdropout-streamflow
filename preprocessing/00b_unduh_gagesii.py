#!/usr/bin/env python3
"""
Unduh raw GAGES-II, lalu ambil semua atributnya untuk pos yang diminta.

Keluaran:
  raw/gagesii/basinchar.zip           arsip asli dari ScienceBase
  raw/gagesii/gagesii_atribut.csv     semua atribut pos terpilih
  raw/gagesii/gagesii_var_desc.csv    arti tiap variabel
  raw/gagesii/gagesii_lembar.csv      daftar lembar dan jumlah atributnya

Cara pakai:
  python3 00b_unduh_gagesii.py 02035000 02019500 ... --output raw/gagesii
"""
import argparse
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

# basinchar_and_report_sept_2011.zip di ScienceBase (55 MB)
URL_ZIP = ("https://www.sciencebase.gov/catalog/file/get/"
           "631405bbd34e36012efa304a?f=__disk__01%2F0a%2Fd2%2F"
           "010ad2fb628d42c764b81596fda93f0e86be478c")

# lembar administratif, bukan atribut fisik
LEWATI = {"X_Region_Names", "Bound_QA"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pos", nargs="+", help="kode pos USGS")
    ap.add_argument("--output", default="raw/gagesii")
    a = ap.parse_args()

    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    zp = out / "basinchar.zip"

    if not zp.exists():
        print("mengunduh GAGES-II dari ScienceBase (55 MB) ...")
        r = requests.get(URL_ZIP, timeout=900)
        r.raise_for_status()
        zp.write_bytes(r.content)
        print(f"  tersimpan: {zp} ({zp.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"arsip sudah ada: {zp}")

    with zipfile.ZipFile(zp) as z:
        nama = [n for n in z.namelist() if n.endswith(".xlsx")]
        utama = [n for n in nama if "conterm" in n][0]
        desc = [n for n in nama if "var_desc" in n][0]
        z.extract(utama, out)
        z.extract(desc, out)

    fx = out / utama
    print(f"\nmembaca {Path(utama).name} ...")
    x = pd.ExcelFile(fx)

    pos = [str(p).zfill(8) for p in a.pos]
    gabung = None
    ringkas = []

    for s in x.sheet_names:
        d = x.parse(s)
        kol_id = next((c for c in d.columns
                       if str(c).upper() in ("STAID", "GAGE_ID", "SITE_NO")), None)
        if kol_id is None:
            ringkas.append({"lembar": s, "atribut": len(d.columns),
                            "status": "dilewati, tanpa kolom ID"})
            continue
        if s in LEWATI:
            ringkas.append({"lembar": s, "atribut": len(d.columns) - 1,
                            "status": "dilewati, administratif"})
            continue

        d[kol_id] = d[kol_id].astype(str).str.strip().str.zfill(8)
        d = d[d[kol_id].isin(pos)].copy()
        d = d.rename(columns={kol_id: "STAID"})
        # kasih awalan nama lembar biar kolomnya tidak bentrok
        d = d.rename(columns={c: f"{s}__{c}" for c in d.columns if c != "STAID"})
        ringkas.append({"lembar": s, "atribut": len(d.columns) - 1,
                        "status": f"{len(d)} pos ditemukan"})
        gabung = d if gabung is None else gabung.merge(d, on="STAID", how="outer")

    if gabung is None or gabung.empty:
        print("Tidak ada pos yang cocok di GAGES-II.")
        return 1

    gabung = gabung.set_index("STAID").reindex(pos).reset_index()
    gabung.to_csv(out / "gagesii_atribut.csv", index=False)

    dd = x.parse(Path(desc).name.replace(".xlsx", "")) if False else \
        pd.read_excel(out / desc)
    dd.to_csv(out / "gagesii_var_desc.csv", index=False)

    rs = pd.DataFrame(ringkas)
    rs.to_csv(out / "gagesii_lembar.csv", index=False)

    print("\n=== lembar GAGES-II ===")
    print(rs.to_string(index=False))
    print(f"\npos diminta      : {len(pos)}")
    print(f"pos ditemukan    : {gabung.notna().any(axis=1).sum()}")
    print(f"atribut tergabung: {len(gabung.columns) - 1}")
    print(f"\nDisimpan: {out / 'gagesii_atribut.csv'}")

    hilang = [p for p in pos
              if gabung.loc[gabung['STAID'] == p].drop(columns='STAID').isna().all(axis=1).all()]
    if hilang:
        print(f"\nTidak ada di GAGES-II: {hilang}")
        print("(GAGES-II hanya memuat pos dengan rekaman minimal 20 tahun sejak 1950)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
