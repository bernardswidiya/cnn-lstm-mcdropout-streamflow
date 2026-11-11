#!/usr/bin/env python3
"""
Unduh raw data debit harian dan jaringan sungai dari USGS.

Keluaran:
  raw/network/nldi_upstream_02035000.json   pos-pos di hulu (NLDI)
  raw/sites/site_info.rdb                   metadata pos
  raw/sites/series_catalog.rdb              katalog deret waktu
  raw/discharge/dv_<kode>.rdb               debit harian per pos
"""
import argparse
import sys
import time
from pathlib import Path

import requests

NLDI = "https://api.water.usgs.gov/nldi/linked-data/nwissite"
NWIS_SITE = "https://waterservices.usgs.gov/nwis/site/"
NWIS_DV = "https://waterservices.usgs.gov/nwis/dv/"

PAR_DEBIT = "00060"
STAT_RATA = "00003"


def simpan(path: Path, teks: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(teks, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--muara", default="02035000",
                    help="kode pos muara, titik awal penelusuran ke hulu")
    ap.add_argument("--jarak", type=int, default=300,
                    help="radius penelusuran ke hulu, km")
    ap.add_argument("--mulai", default="2000-01-01")
    ap.add_argument("--selesai", default="2024-12-31")
    ap.add_argument("--output", default="raw")
    a = ap.parse_args()

    out = Path(a.output)

    # 1. NLDI: cari pos di hulu
    url = (f"{NLDI}/USGS-{a.muara}/navigation/UT/nwissite"
           f"?f=json&distance={a.jarak}")
    print(f"[1/4] NLDI: menelusuri ke hulu dari {a.muara} ...")
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    simpan(out / "network" / f"nldi_upstream_{a.muara}.json", r.text)

    kode = sorted({f["properties"]["identifier"].replace("USGS-", "")
                   for f in r.json().get("features", [])})
    print(f"      {len(kode)} pos terhubung")

    # 2. metadata pos
    print("[2/4] NWIS site: metadata seluruh pos ...")
    bagian = []
    for i in range(0, len(kode), 100):
        p = {"format": "rdb", "sites": ",".join(kode[i:i + 100]),
             "siteOutput": "expanded", "siteStatus": "all"}
        rr = requests.get(NWIS_SITE, params=p, timeout=300)
        rr.raise_for_status()
        bagian.append(rr.text)
        time.sleep(0.4)
    simpan(out / "sites" / "site_info.rdb", "\n".join(bagian))

    # 3. katalog deret waktu
    print("[3/4] NWIS site: katalog deret waktu ...")
    bagian = []
    for i in range(0, len(kode), 100):
        p = {"format": "rdb", "sites": ",".join(kode[i:i + 100]),
             "seriesCatalogOutput": "true", "siteStatus": "all"}
        rr = requests.get(NWIS_SITE, params=p, timeout=300)
        rr.raise_for_status()
        bagian.append(rr.text)
        time.sleep(0.4)
    simpan(out / "sites" / "series_catalog.rdb", "\n".join(bagian))

    # 4. debit harian per pos
    print(f"[4/4] NWIS dv: debit harian {a.mulai} s.d. {a.selesai} ...")
    ada, kosong = [], []
    for n, k in enumerate(kode, 1):
        p = {"format": "rdb", "sites": k, "parameterCd": PAR_DEBIT,
             "statCd": STAT_RATA, "startDT": a.mulai, "endDT": a.selesai,
             "siteStatus": "all"}
        try:
            rr = requests.get(NWIS_DV, params=p, timeout=300)
        except Exception as e:
            print(f"      [{n}/{len(kode)}] {k}: GAGAL {e}")
            continue
        if rr.status_code == 404 or "No sites found" in rr.text:
            kosong.append(k)
            print(f"      [{n}/{len(kode)}] {k}: tidak ada deret harian")
            time.sleep(0.3)
            continue
        rr.raise_for_status()
        isi = [x for x in rr.text.splitlines()
               if x and not x.startswith("#")]
        if len(isi) < 3:
            kosong.append(k)
            print(f"      [{n}/{len(kode)}] {k}: kosong")
            time.sleep(0.3)
            continue
        simpan(out / "discharge" / f"dv_{k}.rdb", rr.text)
        ada.append(k)
        print(f"      [{n}/{len(kode)}] {k}: {len(isi) - 2} baris")
        time.sleep(0.3)

    print(f"\nSelesai.")
    print(f"  pos terhubung          : {len(kode)}")
    print(f"  punya debit harian     : {len(ada)}")
    print(f"  tanpa debit harian     : {len(kosong)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
