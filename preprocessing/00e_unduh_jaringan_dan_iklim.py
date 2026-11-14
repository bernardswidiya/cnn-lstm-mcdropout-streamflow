#!/usr/bin/env python3
"""
Unduh data jaringan sungai, poligon DAS, iklim harian, dan DEM untuk 23 pos debit.

Semua keluaran masuk ke file atau folder baru. File lama di raw/ tidak disentuh,
dan file yang sudah ada tidak ditimpa kecuali diberi --timpa.

Keluaran:
  raw/network/dm/<kode>.json                 navigasi hilir (DM, flowlines, 600 km)
  raw/network/basin/<kode>.json              poligon DAS (NLDI, tidak disederhanakan)
  raw/network/peta/ut_02035000.json          seluruh alur hulu dari outlet (UT, 600 km)
  raw/network/peta/nhdplus_vaa_james.csv     NHDPlus V2 VAA untuk COMID alur UT
  raw/iklim/cek_kabisat_daymet.csv           cek 31 Desember tahun kabisat (Daymet single-pixel)
  raw/iklim/daymet_<kode>.csv                P, Tmax, Tmin rata-rata poligon DAS (Daymet V4 R1)
  raw/iklim/daymet_19991231.csv              Daymet 31 Desember 1999 untuk semua pos (hari sebelum periode)
  raw/iklim/gridmet_isi_kabisat.csv          gridMET untuk 31 Desember tahun kabisat
  raw/spatial/dem/<kode>.tif                 DEM 3DEP 90 m per poligon DAS

Daymet V4 R1 diambil dari server OPeNDAP NASA Earthdata (Hyrax), dengan subset
indeks x-y di sekitar seluruh poligon DAS. Layanan NCSS THREDDS ORNL yang dipakai
pydaymet sudah tidak tersedia (HTTP 404 sejak November 2026), sehingga pydaymet
tidak dipakai. Server ini memerlukan akun NASA Earthdata. Isi ~/.netrc dengan baris
  machine urs.earthdata.nasa.gov login <user> password <sandi>
sebelum langkah Daymet dijalankan. Tanpa itu langkah Daymet dilewati.
"""
import argparse
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# cache HyRiver di folder sementara, bukan di repo
os.environ.setdefault("HYRIVER_CACHE_NAME", str(Path(tempfile.gettempdir()) / "hyriver_cache.sqlite"))

NLDI = "https://api.water.usgs.gov/nldi/linked-data/nwissite"
DAYMET_PIKSEL = "https://daymet.ornl.gov/single-pixel/api/data"
# koleksi Daymet_Daily_V4R1_2129 di CMR (C2532426483-ORNL_CLOUD)
DAYMET_DAP = ("https://opendap.earthdata.nasa.gov/collections/C2532426483-ORNL_CLOUD/granules/"
              "Daymet_Daily_V4R1.daymet_v4_daily_na_{var}_{tahun}.nc.dap.nc4")
TAHUN_KABISAT = [2000, 2004, 2008, 2012, 2016, 2020, 2024]


def ambil(url: str, coba: int = 4) -> str:
    for i in range(coba):
        try:
            r = requests.get(url, timeout=600)
            r.raise_for_status()
            return r.text
        except Exception as e:
            if i == coba - 1:
                raise
            print(f"      ulang ({e.__class__.__name__}) ...")
            time.sleep(5 * (i + 1))


def simpan(path: Path, teks: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(teks, encoding="utf-8")


def perlu(path: Path, timpa: bool) -> bool:
    return timpa or not path.exists()


def daftar_pos(raw: Path) -> list[str]:
    return sorted(p.stem.replace("dv_", "") for p in (raw / "discharge").glob("dv_*.rdb"))


def baca_poligon(raw: Path, pos: list[str]):
    import geopandas as gpd
    g = [gpd.read_file(raw / "network" / "basin" / f"{k}.json").assign(pos=k) for k in pos]
    return gpd.GeoDataFrame(pd.concat(g, ignore_index=True), crs=4326)[["pos", "geometry"]]


def rata_poligon(da, poligon) -> pd.DataFrame:
    """Rata-rata sel grid yang pusatnya jatuh di dalam tiap poligon (kolom = pos).
    Poligon yang tidak memuat satu pun pusat sel memakai semua sel yang tersentuh."""
    from rasterio.features import geometry_mask
    pol = poligon.to_crs(da.rio.crs)
    bentuk = (da.sizes[da.rio.y_dim], da.sizes[da.rio.x_dim])
    hasil = {}
    for k, geom in zip(pol["pos"], pol.geometry):
        m = ~geometry_mask([geom], bentuk, da.rio.transform(), all_touched=False)
        if m.sum() == 0:
            m = ~geometry_mask([geom], bentuk, da.rio.transform(), all_touched=True)
        nilai = da.values.reshape(da.shape[0], -1)[:, m.ravel()]
        hasil[k] = np.nanmean(nilai, axis=1)
    return pd.DataFrame(hasil, index=pd.to_datetime(da["time"].values))


def ada_kredensial_earthdata() -> bool:
    for nama in (".netrc", "_netrc"):
        f = Path.home() / nama
        if f.exists() and "urs.earthdata.nasa.gov" in f.read_text():
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--muara", default="02035000")
    ap.add_argument("--jarak", type=int, default=600, help="jarak navigasi NLDI, km")
    ap.add_argument("--mulai", default="2000-01-01")
    ap.add_argument("--selesai", default="2024-12-31")
    ap.add_argument("--resolusi-dem", type=int, default=90, help="resolusi DEM, m")
    ap.add_argument("--output", default="raw")
    ap.add_argument("--langkah", default="1234567",
                    help="langkah yang dijalankan, misalnya 123 untuk jaringan saja")
    ap.add_argument("--timpa", action="store_true", help="timpa file yang sudah ada")
    a = ap.parse_args()

    raw = Path(a.output)
    pos = daftar_pos(raw)
    print(f"{len(pos)} pos debit: {', '.join(pos)}\n")

    # 1. navigasi hilir per pos
    if "1" in a.langkah:
        print(f"[1/7] NLDI DM flowlines {a.jarak} km per pos ...")
        for k in pos:
            f = raw / "network" / "dm" / f"{k}.json"
            if perlu(f, a.timpa):
                simpan(f, ambil(f"{NLDI}/USGS-{k}/navigation/DM/flowlines?distance={a.jarak}"))
                time.sleep(0.3)
            print(f"      {k}: {len(json.loads(f.read_text())['features'])} ruas")

    # 2. poligon DAS per pos
    if "2" in a.langkah:
        print("[2/7] NLDI basin per pos (simplified=false) ...")
        for k in pos:
            f = raw / "network" / "basin" / f"{k}.json"
            if perlu(f, a.timpa):
                simpan(f, ambil(f"{NLDI}/USGS-{k}/basin?simplified=false"))
                time.sleep(0.3)
        print(f"      {len(pos)} poligon")

    # 3. seluruh alur hulu dari outlet, untuk peta
    if "3" in a.langkah:
        f = raw / "network" / "peta" / f"ut_{a.muara}.json"
        print(f"[3/7] NLDI UT flowlines {a.jarak} km dari {a.muara} ...")
        if perlu(f, a.timpa):
            simpan(f, ambil(f"{NLDI}/USGS-{a.muara}/navigation/UT/flowlines?distance={a.jarak}"))
        print(f"      {len(json.loads(f.read_text())['features'])} ruas")

    # 4. NHDPlus V2 VAA (orde sungai) untuk COMID alur UT
    if "4" in a.langkah:
        f = raw / "network" / "peta" / "nhdplus_vaa_james.csv"
        print("[4/7] NHDPlus V2 VAA (pynhd.nhdplus_vaa) ...")
        if perlu(f, a.timpa):
            import pynhd
            ut = json.loads((raw / "network" / "peta" / f"ut_{a.muara}.json").read_text())
            comid = {int(x["properties"]["nhdplus_comid"]) for x in ut["features"]}
            vaa = pynhd.nhdplus_vaa(Path(tempfile.gettempdir()) / "nhdplus_vaa.parquet")
            vaa = vaa[vaa["comid"].astype("int64").isin(comid)].sort_values("comid")
            f.parent.mkdir(parents=True, exist_ok=True)
            vaa.to_csv(f, index=False)
            print(f"      {len(vaa)} dari {len(comid)} COMID ditemukan di VAA")
        else:
            print(f"      sudah ada: {f}")

    # 5. cek 31 Desember tahun kabisat di Daymet (single-pixel API, titik pos muara)
    hilang_kabisat = None
    if "5" in a.langkah or "6" in a.langkah:
        f = raw / "iklim" / "cek_kabisat_daymet.csv"
        print("[5/7] Cek 31 Desember tahun kabisat di Daymet ...")
        if perlu(f, a.timpa):
            info = json.loads(ambil(f"{NLDI}/USGS-{a.muara}"))
            bujur, lintang = info["features"][0]["geometry"]["coordinates"]
            bagian = []
            for th in TAHUN_KABISAT:
                akhir = min(th + 1, 2024)
                teks = ambil(f"{DAYMET_PIKSEL}?lat={lintang}&lon={bujur}&vars=prcp"
                             f"&start={th}-12-28&end={akhir}-{'01-01' if akhir > th else '12-31'}")
                isi = teks[teks.index("year,yday"):]
                bagian.append(pd.read_csv(io.StringIO(isi)))
                time.sleep(0.5)
            simpan(f, pd.concat(bagian, ignore_index=True).drop_duplicates().to_csv(index=False))
        cek = pd.read_csv(f)
        yday_maks = cek[cek["year"].isin(TAHUN_KABISAT)].groupby("year")["yday"].max()
        hilang_kabisat = bool((yday_maks == 365).all())
        print(f"      yday terbesar di tahun kabisat: {yday_maks.to_dict()}")
        print(f"      31 Desember tahun kabisat {'HILANG' if hilang_kabisat else 'ADA'} di Daymet")

    # 6. Daymet + gridMET rata-rata poligon
    if "6" in a.langkah:
        poligon = baca_poligon(raw, pos)
        kotak = tuple(poligon.total_bounds)
        th0, th1 = int(a.mulai[:4]), int(a.selesai[:4])

        print("[6/7] Daymet V4 R1 (OPeNDAP NASA Earthdata) rata-rata poligon ...")
        target = [raw / "iklim" / f"daymet_{k}.csv" for k in pos]
        # hari sebelum periode, dipakai sebagai nilai hari pertama setelah input Daymet digeser +1 hari
        f_sebelum = raw / "iklim" / f"daymet_{th0 - 1}1231.csv"
        perlu_seri = any(perlu(f, a.timpa) for f in target)
        perlu_sebelum = perlu(f_sebelum, a.timpa)
        if not (perlu_seri or perlu_sebelum):
            print("      sudah ada semua")
        elif not ada_kredensial_earthdata():
            print("      DILEWATI: OPeNDAP Daymet memerlukan login NASA Earthdata, dan ~/.netrc")
            print("      belum memuat urs.earthdata.nasa.gov. Lihat docstring skrip ini.")
        else:
            import xarray as xr
            from pyproj import CRS
            sesi = requests.Session()             # kredensial dibaca dari ~/.netrc

            def ambil_nc(url):
                for i in range(4):
                    try:
                        r = sesi.get(url, timeout=900)
                        r.raise_for_status()
                        with tempfile.NamedTemporaryFile(suffix=".nc4") as tmp:
                            Path(tmp.name).write_bytes(r.content)
                            return xr.open_dataset(tmp.name).load()
                    except Exception as e:
                        if i == 3:
                            raise
                        print(f"      ulang ({e.__class__.__name__}) ...")
                        time.sleep(10 * (i + 1))

            # koordinat grid dan proyeksi dari satu granul, lalu indeks kotak di sekitar semua poligon
            dasar = DAYMET_DAP.format(var="prcp", tahun=th0)
            xy = ambil_nc(f"{dasar}?dap4.ce=/x;/y;/lambert_conformal_conic")
            crs = CRS.from_cf(xy["lambert_conformal_conic"].attrs)
            bx0, by0, bx1, by1 = poligon.to_crs(crs).total_bounds
            x, y = xy["x"].values, xy["y"].values
            jx = np.where((x >= bx0 - 3000) & (x <= bx1 + 3000))[0]
            iy = np.where((y >= by0 - 3000) & (y <= by1 + 3000))[0]
            i0, i1, j0, j1 = iy.min(), iy.max(), jx.min(), jx.max()
            print(f"      subset grid: {i1 - i0 + 1} x {j1 - j0 + 1} sel")

            def rata_daymet(v, th, t0, t1):
                """Rata-rata poligon variabel v tahun th, indeks hari t0..t1 (0 = 1 Januari)."""
                ds = ambil_nc(f"{DAYMET_DAP.format(var=v, tahun=th)}?dap4.ce=/{v}[{t0}:1:{t1}][{i0}:1:{i1}][{j0}:1:{j1}];"
                              f"/x[{j0}:1:{j1}];/y[{i0}:1:{i1}];/time[{t0}:1:{t1}]")
                da = ds[v].transpose("time", "y", "x").rio.set_spatial_dims(x_dim="x", y_dim="y")
                return rata_poligon(da.rio.write_crs(crs), poligon)

            if perlu_seri:
                per_var = {v: [] for v in ("prcp", "tmax", "tmin")}
                for th in range(th0, th1 + 1):
                    for v in per_var:
                        per_var[v].append(rata_daymet(v, th, 0, 364))
                    print(f"      {th}: {len(per_var['prcp'][-1])} hari")
                per_var = {v: pd.concat(x) for v, x in per_var.items()}
                for k, f in zip(pos, target):
                    d = pd.DataFrame({v: per_var[v][k] for v in per_var})
                    d.index.name = "time"
                    f.parent.mkdir(parents=True, exist_ok=True)
                    d.to_csv(f, float_format="%.3f")
                print(f"      {len(pos)} file daymet_<kode>.csv")

            if perlu_sebelum:
                # 31 Desember tahun sebelum periode; tahun itu bukan tahun kabisat sehingga indeksnya 364
                assert (th0 - 1) % 4 != 0, "tahun sebelum periode adalah tahun kabisat"
                satu = {v: rata_daymet(v, th0 - 1, 364, 364) for v in ("prcp", "tmax", "tmin")}
                d = pd.DataFrame({"time": [satu["prcp"].index[0]] * len(pos), "pos": pos,
                                  **{v: [float(satu[v][k].iloc[0]) for k in pos] for v in satu}})
                assert str(pd.Timestamp(d["time"].iloc[0]).date()) == f"{th0 - 1}-12-31"
                f_sebelum.parent.mkdir(parents=True, exist_ok=True)
                d.to_csv(f_sebelum, index=False, float_format="%.3f")
                print(f"      {f_sebelum.name}: {len(pos)} pos")

        f = raw / "iklim" / "gridmet_isi_kabisat.csv"
        print("      gridMET (NCSS, alamat dari pygridmet) untuk 31 Desember tahun kabisat ...")
        if hilang_kabisat is False:
            print("      dilewati: 31 Desember kabisat ada di Daymet")
        elif not perlu(f, a.timpa):
            print(f"      sudah ada: {f}")
        else:
            # pygridmet sengaja membuang 31 Desember tahun kabisat (mengikuti kalender
            # 365 hari Daymet), jadi layanan NCSS gridMET yang sama dipanggil langsung
            # dengan alamat dan nama variabel milik pygridmet.
            import xarray as xr
            from pygridmet.core import GridMET
            from pygridmet.pygridmet import URL as GRIDMET_NCSS
            nama_panjang = GridMET(2000, ["pr"], False).long_names
            barat, selatan, timur, utara = kotak
            baris = []
            for th in [t for t in TAHUN_KABISAT if th0 <= t <= th1]:
                tgl = f"{th}-12-31"
                rata = {}
                for v in ("pr", "tmmx", "tmmn"):
                    p = {"var": nama_panjang[v], "north": utara + 0.05, "south": selatan - 0.05,
                         "west": barat - 0.05, "east": timur + 0.05, "disableProjSubset": "on",
                         "horizStride": 1, "time_start": f"{tgl}T00:00:00Z",
                         "time_end": f"{tgl}T00:00:00Z", "timeStride": 1, "addLatLon": "true",
                         "accept": "netcdf"}
                    r = requests.get(f"{GRIDMET_NCSS}/agg_met_{v}_1979_CurrentYear_CONUS.nc",
                                     params=p, timeout=600)
                    r.raise_for_status()
                    with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
                        Path(tmp.name).write_bytes(r.content)
                        da = xr.open_dataset(tmp.name)[nama_panjang[v]].load()
                    da = (da.rename(day="time").rio.set_spatial_dims(x_dim="lon", y_dim="lat")
                          .rio.write_crs(4326))
                    assert str(pd.Timestamp(da["time"].values[0]).date()) == tgl
                    rata[v] = rata_poligon(da, poligon)
                for k in pos:
                    baris.append({"time": tgl, "pos": k,
                                  **{v: float(rata[v][k].iloc[0]) for v in rata}})
                print(f"      {tgl}")
            simpan(f, pd.DataFrame(baris).to_csv(index=False, float_format="%.3f"))

    # 7. DEM 3DEP per poligon
    if "7" in a.langkah:
        import py3dep
        poligon = baca_poligon(raw, pos)
        print(f"[7/7] DEM 3DEP {a.resolusi_dem} m per poligon (py3dep) ...")
        for k, geom in zip(poligon["pos"], poligon.geometry):
            f = raw / "spatial" / "dem" / f"{k}.tif"
            if perlu(f, a.timpa):
                dem = py3dep.get_dem(geom, a.resolusi_dem, crs=4326)
                f.parent.mkdir(parents=True, exist_ok=True)
                dem.rio.to_raster(f, compress="deflate")
            print(f"      {k}")

    print("\nSelesai.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
