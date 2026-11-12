#!/usr/bin/env python3
"""
Hitung atribut spasial tiap pos dari data USGS.

  NWIS site      : elevasi, luas DAS, koordinat, HUC
  NLDI basin     : poligon DAS -> luas, keliling, kebulatan
  NLDI flowlines : alur sungai -> jarak ke muara, kemiringan

Keluaran: raw/spatial/atribut_turunan.csv (lewat opsi -o; bawaan atribut_spasial.csv)
"""
import sys, time, math, json
import numpy as np, pandas as pd, requests

NLDI = "https://api.water.usgs.gov/nldi/linked-data/nwissite"
SITE = "https://waterservices.usgs.gov/nwis/site/"
R_BUMI = 6371008.8          # meter
FT_KE_M = 0.3048
MI2_KE_KM2 = 2.58999


def haversine(a, b):
    """Jarak dua titik (bujur, lintang), meter."""
    l1, t1, l2, t2 = map(math.radians, [a[0], a[1], b[0], b[1]])
    h = (math.sin((t2 - t1) / 2) ** 2
         + math.cos(t1) * math.cos(t2) * math.sin((l2 - l1) / 2) ** 2)
    return 2 * R_BUMI * math.asin(min(1.0, math.sqrt(h)))


def panjang_garis(koord):
    return sum(haversine(koord[i], koord[i + 1]) for i in range(len(koord) - 1))


def luas_poligon(ring):
    """Luas poligon (bujur, lintang) dalam m2, pakai proyeksi sama-luas sederhana."""
    t0 = math.radians(np.mean([p[1] for p in ring]))
    xy = [(math.radians(p[0]) * R_BUMI * math.cos(t0),
           math.radians(p[1]) * R_BUMI) for p in ring]
    s = sum(xy[i][0] * xy[i + 1][1] - xy[i + 1][0] * xy[i][1]
            for i in range(len(xy) - 1))
    return abs(s) / 2.0


def perimeter_poligon(ring):
    return panjang_garis(ring)


# NWIS
def nwis_atribut(kode):
    out = {}
    for i in range(0, len(kode), 100):
        bag = kode[i:i + 100]
        r = requests.get(SITE, params={
            "format": "rdb", "sites": ",".join(bag),
            "siteOutput": "expanded", "siteStatus": "all"}, timeout=120)
        b = [x for x in r.text.splitlines() if x and not x.startswith("#")]
        if len(b) < 3:
            continue
        kol = b[0].split("\t")
        for ln in b[2:]:
            v = ln.split("\t")
            if len(v) != len(kol):
                continue
            d = dict(zip(kol, v))
            alt = pd.to_numeric(d.get("alt_va"), errors="coerce")
            da = pd.to_numeric(d.get("drain_area_va"), errors="coerce")
            out[d["site_no"]] = {
                "nama": d.get("station_nm", "").strip(),
                "lintang": pd.to_numeric(d.get("dec_lat_va"), errors="coerce"),
                "bujur": pd.to_numeric(d.get("dec_long_va"), errors="coerce"),
                "elevasi_m": alt * FT_KE_M if pd.notna(alt) else np.nan,
                "luas_nwis_km2": da * MI2_KE_KM2 if pd.notna(da) else np.nan,
                "huc": d.get("huc_cd", ""),
            }
        time.sleep(0.4)
    return out


# NLDI basin
def basin_atribut(kode):
    r = requests.get(f"{NLDI}/USGS-{kode}/basin", params={"f": "json"},
                     timeout=120)
    if r.status_code != 200:
        return {}
    f = r.json().get("features", [])
    if not f:
        return {}
    g = f[0]["geometry"]
    ring = (g["coordinates"][0][0] if g["type"] == "MultiPolygon"
            else g["coordinates"][0])
    luas = luas_poligon(ring) / 1e6
    peri = perimeter_poligon(ring) / 1e3
    lat = [p[1] for p in ring]
    lon = [p[0] for p in ring]
    return {
        "luas_poligon_km2": round(luas, 1),
        "perimeter_km": round(peri, 1),
        # kebulatan: 1 = bulat, makin kecil makin lonjong
        "kebulatan": round(4 * math.pi * luas / (peri ** 2), 3) if peri else np.nan,
        "rentang_lintang_km": round((max(lat) - min(lat)) * 111.32, 1),
        "rentang_bujur_km": round((max(lon) - min(lon)) * 111.32
                                  * math.cos(math.radians(np.mean(lat))), 1),
    }


# jarak alur
def alur_dm(kode, jarak=800):
    """Segmen alur di hilir pos beserta panjangnya."""
    r = requests.get(f"{NLDI}/USGS-{kode}/navigation/DM/flowlines",
                     params={"f": "json", "distance": jarak}, timeout=180)
    r.raise_for_status()
    d = {}
    for f in r.json().get("features", []):
        cid = f["properties"].get("nhdplus_comid")
        g = f["geometry"]
        koord = (g["coordinates"] if g["type"] == "LineString"
                 else [p for part in g["coordinates"] for p in part])
        d[cid] = panjang_garis(koord)
    return d


def jarak_ke_muara(kode_list, muara):
    """Panjang alur dari tiap pos ke muara, km."""
    alur = {}
    for k in kode_list:
        alur[k] = alur_dm(k)
        print(f"  alur {k}: {len(alur[k])} segmen", flush=True)
        time.sleep(0.6)
    set_muara = set(alur[muara])
    out = {}
    for k in kode_list:
        antara = set(alur[k]) - set_muara
        out[k] = round(sum(alur[k][c] for c in antara) / 1000.0, 1)
    return out


if __name__ == "__main__":
    arg = sys.argv[1:]
    output = "atribut_spasial.csv"
    if "-o" in arg:
        i = arg.index("-o")
        output = arg[i + 1]
        arg = arg[:i] + arg[i + 2:]
    kode = arg[:-1]
    muara = arg[-1]
    if muara not in kode:
        kode.append(muara)

    print("1/3 atribut NWIS ...")
    nw = nwis_atribut(kode)

    print("2/3 poligon DAS ...")
    bs = {}
    for k in kode:
        bs[k] = basin_atribut(k)
        print(f"  basin {k}: {bs[k].get('luas_poligon_km2','gagal')} km2", flush=True)
        time.sleep(0.6)

    print("3/3 jarak alur ke muara ...")
    jr = jarak_ke_muara(kode, muara)

    baris = []
    for k in kode:
        d = {"kode_pos": k}
        d.update(nw.get(k, {}))
        d.update(bs.get(k, {}))
        d["jarak_alur_ke_muara_km"] = jr.get(k)
        baris.append(d)

    df = pd.DataFrame(baris).sort_values("jarak_alur_ke_muara_km",
                                         ascending=False).reset_index(drop=True)

    # kemiringan dasar sungai dari pos ke muara
    e_muara = df.loc[df["kode_pos"] == muara, "elevasi_m"]
    e_muara = float(e_muara.iloc[0]) if len(e_muara) else np.nan
    df["kemiringan_ke_muara_m_per_km"] = (
        (df["elevasi_m"] - e_muara) / df["jarak_alur_ke_muara_km"]).round(3)

    df.to_csv(output, index=False)
    pd.set_option("display.width", 220)
    print("\n=== Atribut spasial (urut dari hulu ke hilir) ===")
    print(df[["kode_pos", "elevasi_m", "luas_nwis_km2", "luas_poligon_km2",
              "kebulatan", "jarak_alur_ke_muara_km",
              "kemiringan_ke_muara_m_per_km"]].to_string(index=False))
    print(f"\nDisimpan: {output}")
