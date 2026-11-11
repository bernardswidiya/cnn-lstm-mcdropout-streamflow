#!/usr/bin/env python3
"""
Unduh raw data hujan harian GHCN-Daily (NOAA).

Keluaran:
  raw/hujan/ghcnd-readme.txt       penjelasan format .dly
  raw/hujan/ghcnd-stations.txt     daftar stasiun GHCN
  raw/hujan/ghcnd-inventory.txt    elemen dan rentang tahun tiap stasiun
  raw/hujan/dly/<ID>.dly           data per stasiun
"""
import sys, time
from pathlib import Path
import requests

BASE = "https://www.ncei.noaa.gov/pub/data/ghcn/daily"

# 16 stasiun di sekitar DAS James, data PRCP >= 95% lengkap (2000-2024)
STASIUN = ["USW00013733","USW00013741","USC00443375","USC00443310","USC00447278",
           "USC00441955","USC00445595","USC00446491","USW00093736","USC00445050",
           "USC00442044","USC00445880","USC00442142","USC00468308","USC00446712",
           "USC00448941"]


def simpan(path, isi):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(isi)


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "raw/hujan")

    print("[1/2] file pendukung ...")
    for n in ["readme.txt", "ghcnd-stations.txt", "ghcnd-inventory.txt"]:
        u = f"{BASE}/{n}"
        r = requests.get(u, timeout=600); r.raise_for_status()
        nama = "ghcnd-readme.txt" if n == "readme.txt" else n
        simpan(out / nama, r.content)
        print(f"      {nama}: {len(r.content)/1e6:.1f} MB")

    print(f"[2/2] {len(STASIUN)} stasiun, format .dly ...")
    ada = []
    for i, s in enumerate(STASIUN, 1):
        u = f"{BASE}/all/{s}.dly"
        try:
            r = requests.get(u, timeout=300); r.raise_for_status()
        except Exception as e:
            print(f"      [{i}/{len(STASIUN)}] {s}: GAGAL {e}"); continue
        simpan(out / "dly" / f"{s}.dly", r.content)
        ada.append(s)
        print(f"      [{i}/{len(STASIUN)}] {s}: {len(r.content)/1e6:.2f} MB")
        time.sleep(0.3)

    print(f"\nSelesai. {len(ada)} stasiun -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
