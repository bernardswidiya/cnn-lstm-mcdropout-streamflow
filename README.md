# CNN-LSTM + Monte Carlo Dropout untuk Prediksi Debit Air

Prediksi debit air pada jaringan sungai yang saling terhubung, dengan target di
titik hilir dan horizon 1 sampai 3 hari.

Proyek Individu · Teknologi Informasi, DTETI, Universitas Gadjah Mada

---

## Struktur

```
raw/              data mentah hasil unduhan, belum diolah
preprocessing/    skrip pengunduh 00a–00f dan satu notebook preprocessing
processed/        data siap latih
modeling/         notebook pemodelan 01–03, modul pemodelan.py, hasil/ (log run dan tabel)
figur/            gambar yang disimpan oleh notebook
references/       daftar referensi (PDF tidak disimpan di repo)
```

Data di `raw/` disimpan dalam format asli server: nama kolom, satuan, kode
kualitas, dan baris komentar header tetap seperti aslinya, dengan satu file per
pos. Penggabungan, pengisian gap, dan pemilihan pos dilakukan di notebook.

| File | Isi |
|---|---|
| `raw/network/nldi_upstream_02035000.json` | Topologi jaringan, 249 pos terhubung |
| `raw/sites/site_info.rdb` | Metadata 249 pos, 42 kolom |
| `raw/sites/series_catalog.rdb` | Katalog 3.738 deret waktu |
| `raw/discharge/dv_<kode>.rdb` | Debit harian per pos, RDB asli, ft³/s |
| `raw/gagesii/gagesii_atribut.csv` | 571 atribut DAS per pos |
| `raw/gagesii/gagesii_var_desc.csv` | Penjelasan tiap nama variabel |
| `raw/spatial/atribut_turunan.csv` | Jarak alur, kemiringan, bentuk DAS |
| `raw/hujan/dly/<ID>.dly` | Hujan harian GHCN-Daily, format `.dly` asli |
| `raw/hujan/stasiun_terpakai.txt` | Koordinat 16 stasiun hujan, saringan `ghcnd-stations.txt` |
| `raw/hujan/stasiun_v2.txt`, `inventory_v2.txt` | Baris `ghcnd-stations.txt` dan `ghcnd-inventory.txt` untuk 53 kandidat hujan v2 (skrip `00f`) |
| `raw/hujan/hpd/` | COOP Hourly Precipitation Data v2: Gathright Dam, Piedmont, Mill Gap, inventaris, dan readme |
| `raw/hujan/hash_raw_lama.sha256` | SHA-256 file `raw/` sebelum unduhan v2, untuk memeriksa file lama tetap utuh |
| `raw/network/dm/<kode>.json` | Navigasi hilir NLDI (DM, 600 km) per pos |
| `raw/network/basin/<kode>.json` | Poligon DAS NLDI per pos, tidak disederhanakan |
| `raw/network/peta/ut_02035000.json` | Seluruh alur hulu dari pos target (UT, 600 km) |
| `raw/network/peta/nhdplus_vaa_james.csv` | NHDPlus V2 VAA (orde sungai, hydroseq) untuk alur UT |
| `raw/iklim/cek_kabisat_daymet.csv` | Bukti 31 Desember tahun kabisat tidak ada di Daymet |
| `raw/iklim/gridmet_isi_kabisat.csv` | gridMET rata-rata DAS untuk 31 Desember tahun kabisat |
| `raw/spatial/dem/<kode>.tif` | DEM 3DEP per poligon DAS (EPSG:5070) |
| `preprocessing/00a` – `00f` | Pengunduh, satu per sumber data; `00f` untuk hujan v2 (stasiun GHCN tambahan dan HPD) |
| `preprocessing/preprocessing.ipynb` | Seluruh pengolahan beserta gambar dan tabel, bertahap dalam satu notebook |
| `raw/iklim/daymet_<kode>.csv` | Hujan, Tmax, Tmin Daymet V4 R1 rata-rata poligon DAS per pos |
| `raw/iklim/daymet_19991231.csv` | Daymet 31 Desember 1999, untuk posisi pertama input yang digeser |
| `processed/tensor_h{1,2,3}.npz` | Data siap latih per horizon, varian utama V17 |
| `processed/varian/{V21,V18,V17_daymet_asli}/tensor_h{1,2,3}.npz` | Varian pembanding (V21), ablasi Harris Creek (V18), dan sensitivitas Daymet tanpa geser |
| `processed/tensor_hujan_h{1,2,3}.npz` | Hujan stasiun GHCN-Daily per pos (pembanding, tanggal `.dly` tanpa koreksi jam pengamatan), sejajar dengan tensor V17 |
| `processed/debit_isi_m3s.csv`, `flag_isian.csv`, `catatan_isian.csv` | Debit setelah pengisian gap, flag isian, dan metode per pos |
| `processed/iklim/` | Hujan, Tmax, Tmin Daymet per pos setelah 31 Desember kabisat diisi |
| `processed/hujan/` | Hujan harian per stasiun dan per pos, mutu stasiun, `metadata_hujan.json` |
| `processed/v2/` | Hujan v2 (preprocessing Bagian VIII): `tensor_h{1,2,3}.npz` jendela 8 hari dengan `X_dasar` dan `X_hujan`, `varian_v2.json` (varian R0–R5), hujan harian 43 stasiun + 3 HPD, hujan per pos, tabel stasiun dan metadata |
| `processed/mutu_semua_pos.csv` | Pemeriksaan mutu seluruh pos |
| `processed/metadata.json` | Seluruh parameter dan alasannya |

---

## Cakupan unduhan

Penelusuran memakai USGS NLDI mode `UT` (*upstream with tributaries*) dari pos
target `02035000`, radius 300 km. Mode ini mengikuti jaringan sungai, sehingga
yang terambil hanya pos yang airnya benar-benar mengalir ke pos target.

| | |
|---|---|
| Pos terhubung ke jaringan | **249** |
| Punya deret debit harian 2000–2024 | 23 |
| Tanpa deret debit harian 2000–2024 | 226 |
| Punya curah hujan harian | **0** |

Dari 249 pos itu, 245 adalah pos sungai dan 4 pos danau; tidak ada mata air
maupun sumur. Di antara 226 pos tanpa deret debit 2000–2024, 19 pos punya
rekaman debit harian lama yang berakhir paling lambat 1997.

Karena USGS tidak menyediakan curah hujan di jaringan ini, hujan dan suhu
diambil dari Daymet V4 R1, dirata-rata di poligon DAS setiap pos. Sebagai
pembanding, hujan stasiun diambil dari NOAA GHCN-Daily: 16 stasiun di dalam dan
sekitar DAS. Tujuh di antaranya berkelengkapan di atas 99 persen pada data sah
mentah, dan delapan setelah penambalan.

---

## Dataset

**Sungai James, Virginia, Amerika Serikat.** Sumber USGS dan NOAA, domain publik.

| | |
|---|---|
| Pos terpakai (varian utama V17) | **17** |
| Pos target | 02035000 James River at Cartersville, VA |
| Periode | 1 Januari 2000 – 31 Desember 2024 |
| Jumlah hari | 9.132 |
| Nilai isian (V17) | 127 dari 155.244 sel (0,08 persen), seluruhnya diberi flag |
| Luas DAS target | 16.193 km² |
| Jarak alur terpanjang | 295,9 km |

Aturan pemilihan: rekaman sudah ada sejak 1 Januari 2000. Gap di tengah rekaman
diisi dengan regresi dinamis, metode yang lolos uji gap buatan di tahap 0i
notebook, dan setiap nilai isian diberi flag. Aturan ini menggantikan aturan
lama "gap maksimum 2 hari dengan interpolasi linear", karena uji gap buatan
menunjukkan interpolasi linear kalah sejak gap 1 hari.

Tujuh belas pos, diurutkan dari hulu ke hilir menurut **jarak alur sungai**
(NHDPlus V2); elevasi pos dari DEM 3DEP:

| # | Kode pos | Nama | Elevasi (m) | Luas (km²) | Jarak alur ke pos target (km) |
|---|---|---|---|---|---|
| 1 | 02018000 | Craig Creek at Parr | 302,9 | 852 | 295,9 |
| 2 | 02020500 | Calfpasture River at Goshen | 422,7 | 365 | 282,7 |
| 3 | 02021500 | Maury River at Rockbridge Baths | 334,6 | 852 | 267,0 |
| 4 | 02022500 | Kerrs Creek near Lexington | 298,7 | 91 | 247,7 |
| 5 | 02019500 | James River at Buchanan | 242,3 | 5.369 | 241,8 |
| 6 | 02024000 | Maury River near Buena Vista | 258,5 | 1.676 | 232,6 |
| 7 | 02025500 | James River at Holcomb Rock | 167,8 | 8.433 | 182,1 |
| 8 | 02027500 | Piney River at Piney River | 194,8 | 124 | 132,7 |
| 9 | 02027000 | Tye River near Lovingston | 178,1 | 241 | 132,3 |
| 10 | 02026000 | James River at Bent Creek | 117,8 | 9.451 | 114,6 |
| 11 | 02028500 | Rockfish River near Greenfield | 162,3 | 246 | 111,2 |
| 12 | 02031000 | Mechums River near White Hall | 132,3 | 247 | 109,4 |
| 13 | 02032640 | N.F. Rivanna River near Earlysville | 113,8 | 280 | 98,1 |
| 14 | 02030000 | Hardware River near Scottsville | 92,2 | 300 | 58,4 |
| 15 | 02029000 | James River at Scottsville | 78,9 | 11.865 | 51,1 |
| 16 | 02034000 | Rivanna River at Palmyra | 64,8 | 1.717 | 39,3 |
| 17 | **02035000** | **James River at Cartersville** | 50,3 | 16.193 | 0 (target) |

Varian lain disimpan terpisah dengan pembagian, scaling, dan indeks sampel yang
sama:

| Varian | Pos | Kegunaan |
|---|---|---|
| V21 | V17 + 02024752, 02024915, 02030500, 02032250 (awal rekaman diisi MOVE.1) | Pembanding |
| V18 | V17 + 02025652 Harris Creek (isian menutup 69 persen bagian latih) | Ablasi terpisah |
| V17_daymet_asli | V17, hujan dan Tmax Daymet tanpa geser | Sensitivitas penggeseran Daymet |

02032515 S.F. Rivanna tidak dipakai sama sekali: rekamannya baru mulai Agustus
2020, sehingga parameter isiannya hanya bisa dihitung dari data validasi dan uji.

---

## Data siap latih

```
X : (contoh, 14 langkah waktu, 17 stasiun, 10 saluran)     V21: 21 stasiun, V18: 18 stasiun
y : (contoh,)
```

| Saluran | Isi | Sumber |
|---|---|---|
| 0 | Debit harian, distandarkan | NWIS |
| 1 | Flag isian (1 = nilai isian) | tahap 20 notebook |
| 2 | Hujan rata-rata DAS hari t − 1, log1p lalu distandarkan | Daymet V4 R1 |
| 3 | Tmax rata-rata DAS hari t − 1, distandarkan | Daymet V4 R1 |
| 4 | Tmin rata-rata DAS hari t, distandarkan | Daymet V4 R1 |
| 5 | Luas DAS, dilogaritmakan | poligon NLDI |
| 6 | Elevasi rata-rata DAS (m) | DEM 3DEP |
| 7 | Kemiringan rata-rata DAS (m/km) | DEM 3DEP |
| 8 | Hujan tahunan rata-rata, bagian latih (mm) | Daymet |
| 9 | Indeks aliran dasar, bagian latih | Lyne–Hollick pada debit latih |

Hujan dan Tmax Daymet digeser satu hari karena Daymet V4 memindahkan bacaan pagi
stasiun COOP ke tanggal sebelumnya (Thornton dkk., 2021), sehingga nilai hari t
memuat hujan sekitar 7 jam pertama hari t + 1, yaitu hari target horizon 1.
Posisi 1 Januari 2000 memakai Daymet 31 Desember 1999. Versi tanpa geser
disimpan di `processed/varian/V17_daymet_asli/`.

Saluran 5 sampai 9 bersifat statis, disalin sepanjang sumbu waktu, dan dihitung
dengan cara yang sama untuk semua pos. Lima atribut posisi di jaringan (elevasi
pos, kebulatan, kerapatan alur, jarak alur, kemiringan alur) disimpan terpisah
sebagai `atribut_c` untuk ablasi. Hujan stasiun GHCN-Daily ada di
`tensor_hujan_h*.npz` sebagai pembanding.

Kode bagian: `0` latih, `1` validasi, `2` uji, `3` uji tetapi dikecualikan
karena nilai targetnya hasil perkiraan (`A:e`).

Pengecualian itu hanya berlaku pada nilai yang diprediksi. Jendela masukan
contoh berkode `2` tetap boleh memuat nilai perkiraan: dari 1.322 contoh uji
horizon 1, sebanyak 301 memuat `A:e` pada jendela pos target dan 938 memuat
`A:e` pada pos mana pun. Sebanyak 12 contoh uji memuat satu nilai isian
(02030000, 3 Januari 2022), yang ditandai di saluran 1.

| Horizon | Latih | Validasi | Uji | Uji dikecualikan |
|---|---|---|---|---|
| 1 hari | 6.378 | 1.370 | 1.322 | 48 |
| 2 hari | 6.377 | 1.370 | 1.322 | 48 |
| 3 hari | 6.376 | 1.370 | 1.322 | 48 |

```python
import numpy as np
d = np.load("processed/tensor_h1.npz", allow_pickle=True)
X, y, bagian = d["X"], d["y"], d["bagian"]
X_latih, y_latih = X[bagian == 0], y[bagian == 0]
X_uji,   y_uji   = X[bagian == 2], y[bagian == 2]   # target hasil ukur; jendela masih bisa memuat A:e

# kembalikan ke m3/s
rata, std, it = d["rata"], d["std"], int(d["i_target"])
y_uji_m3s = y_uji * std[it] + rata[it]
```

---

## Keputusan preprocessing

| Keputusan | Pilihan | Alasan |
|---|---|---|
| Periode | 2000–2024 | Satu rezim pengaturan sungai yang mantap |
| Pemilihan pos | Rekaman sejak 1 Januari 2000 (17 pos) | Periode latih tidak dipendekkan; pos yang mulai terlambat hanya di varian pembanding |
| Satuan | m³/s | Dikonversi dari ft³/s, faktor 0,0283168466 |
| Statistik harian | Rata-rata, `statCd=00003` | Lihat catatan API di bawah |
| Gap di tengah rekaman | Regresi dinamis dua donor, parameter dari bagian latih, diberi flag | Lolos uji gap buatan; linear kalah sejak gap 1 hari |
| Gap di bagian latih | Kalman smoother pada data sampai hari terakhir bagian latih (1 Juli 2017) | Isian latih tidak bergantung pada data validasi dan uji |
| Gap di bagian validasi dan uji | Kalman filter, tanpa data sesudah hari itu | Nilai input tidak memuat informasi masa depan |
| Gap sebelum rekaman dimulai (V21, V18) | MOVE.1 log, fit pada tumpang-tindih bagian latih | Bias paling kecil di uji gap buatan |
| Target | Tidak pernah diisi | Nilai isian hanya untuk input |
| Pembagian | 70 : 15 : 15 menurut urutan waktu | Periode uji tidak disentuh sampai akhir |
| Penskalaan | Dihitung dari bagian latih saja, `ddof=0` | Mencegah kebocoran informasi masa depan |
| Transformasi debit | Standardisasi saja | Lihat keterbatasan di bawah |
| Urutan sumbu stasiun | Menurut jarak alur sungai ke pos target | Menyatakan posisi di jaringan |
| Panjang jendela | 14 hari | 1,25 × konstanta waktu resesi, lihat di bawah |
| Horizon | 1, 2, 3 hari | Dilatih terpisah tiap horizon |
| Masukan runtun waktu | Debit, flag, hujan dan Tmax hari t − 1, Tmin hari t | Daymet rata-rata DAS, digeser agar tidak memuat hujan hari target; GHCN-Daily tanpa koreksi jam pengamatan sebagai pembanding |
| Atribut statis | 5, dihitung seragam | GAGES-II hanya mencakup 16 pos dan `BFI_AVE`-nya memuat periode uji |
| Nilai perkiraan `A:e` | Contoh uji bertarget `A:e` ditandai dan dikeluarkan | 3,3 persen hari di pos target |

Urutan **bagi dulu, baru skalakan** tidak boleh dibalik. Kalau parameter
penskalaan dihitung dari seluruh data, informasi periode uji merembes ke
pelatihan dan seluruh hasil evaluasinya cacat.

### Dasar panjang jendela

Dihitung di tahap 23 notebook, memakai **bagian latih saja**:

| Besaran | Nilai | Metode |
|---|---|---|
| Tunda perambatan maksimum | **1 hari** | Korelasi silang pada selisih harian |
| Koefisien resesi harian `k` | **0,9143** | Eksponensial dari median selisih log debit pada seluruh 4.131 hari menurun |
| Konstanta waktu resesi `τ` | **11,2 hari** | −1 / ln `k` |

Jendela 14 hari setara 1,25 × `τ`, mencakup 71 persen sinyal resesi dengan
anggapan peluruhan eksponensial tunggal, sekaligus 14 × tunda perambatan
maksimum. Hari menurun adalah hari ketika debit lebih kecil daripada hari
sebelumnya; tidak ada kejadian resesi yang dipilih secara khusus. Aturan
jendela ≈ 1,25 × `τ` tidak berasal dari literatur, sehingga panjang jendela diuji
ulang di tahap pemodelan (lihat "Uji yang ditunda ke tahap pemodelan").

Resesi melambat seiring umurnya. Pola ini dilaporkan sebagai nuansa, bukan
dasar penetapan jendela:

| Umur resesi | Jumlah hari | Konstanta waktu |
|---|---|---|
| Hari ke 1–7 | 3.523 | 10,1 hari |
| Hari ke 8–14 | 555 | 16,0 hari |
| Hari ke 15 ke atas | 53 | 27,4 hari |

Pekan pertama masih memuat pengatusan aliran cepat; sesudahnya tersisa aliran
dasar dari simpanan air tanah.

Catatan: tunda perambatan 1 hari menyiratkan kecepatan rambat sekitar 3,4 m/s,
lebih cepat daripada angka rujukan umum 1–2 m/s. Penyebabnya kecepatan rambat
gelombang pada sungai sebesar ini memang dapat melampaui 3 m/s pada aliran
tinggi, dan sistem hujan frontal berukuran ratusan kilometer sering menghujani
hulu dan hilir bersamaan.

### Lembar GAGES-II yang tidak dipakai

- **`FlowRec` (115 atribut)** — statistik tentang riwayat alat ukur: 110 kolom
  flag tahun air `wy1900` sampai `wy2009`, jumlah tahun rekaman lengkap
  (`FLOWYRS_*`), persentase nilai perkiraan (`FLOW_PCT_EST_VALUES`), dan status
  aktif 2009 (`ACTIVE09`). Seluruhnya menggambarkan riwayat pemantauan, yaitu
  tahun mana saja pos merekam aliran, bukan perilaku DAS. Untuk 2000 sampai 2009, kesepuluh kolom tahun airnya
  bernilai seragam di ke-16 pos, sehingga tidak membawa informasi pembeda.
  Lembar ini tidak membocorkan periode uji: GAGES-II berhenti pada 2009,
  sedangkan periode uji 2021 sampai 2024.
- **`Climate_Ppt_Annual` dan `Climate_Tmp_Annual`** — nilai per tahun, bukan per
  hari, dan berhenti di 2009.

GAGES-II sekarang hanya dipakai untuk deskripsi (status Ref/Non-ref, derajat
regulasi, fraksi salju). `BFI_AVE` tidak dipakai lagi karena dihitung dari
seluruh rekaman debit, termasuk periode uji. Penggantinya, BFI di saluran 9,
dihitung dari bagian latih saja dengan filter Lyne–Hollick (α = 0,925, tiga
lintasan, pantulan 30 hari; Ladson dkk., 2013), seperti CAMELS.

---

## Menghasilkan ulang

Butuh `pandas`, `numpy`, `matplotlib`, `requests`, `openpyxl`, `jupyter`,
`geopandas`, `rioxarray`, `statsmodels`, serta `pynhd`, `py3dep`, dan
`pygridmet` dari HyRiver. Tahap pemodelan (`modeling/`) menambah `tensorflow`
(2.20, Keras 3). Langkah Daymet memerlukan akun NASA Earthdata yang
ditulis di `~/.netrc` (`machine urs.earthdata.nasa.gov login ... password ...`);
langkah lain tanpa akun.

```bash
# 1. debit + jaringan + metadata  ->  raw/{network,sites,discharge}/
python3 preprocessing/00a_unduh_debit_dan_jaringan.py \
    --muara 02035000 --jarak 300 --mulai 2000-01-01 --selesai 2024-12-31 \
    --output raw

POS="02018000 02019500 02020500 02021500 02022500 02024000 02025500 02026000
     02027000 02027500 02028500 02029000 02030000 02031000 02034000 02035000"

# 2. atribut DAS GAGES-II  ->  raw/gagesii/
python3 preprocessing/00b_unduh_gagesii.py $POS --output raw/gagesii

# 3. atribut spasial turunan (16 pos lama, kini hanya pembanding)  ->  raw/spatial/
python3 preprocessing/00c_atribut_spasial.py $POS -o raw/spatial/atribut_turunan.csv

# 4. curah hujan GHCN-Daily  ->  raw/hujan/
python3 preprocessing/00d_unduh_hujan.py raw/hujan

# 5. navigasi NLDI, poligon DAS, NHDPlus VAA, Daymet, gridMET, DEM 3DEP  ->  raw/{network,iklim,spatial/dem}/
python3 preprocessing/00e_unduh_jaringan_dan_iklim.py

# 6. preprocessing  ->  processed/, figur/
jupyter nbconvert --to notebook --execute --inplace preprocessing/preprocessing.ipynb
```

Langkah 1 sampai 5 hanya perlu dijalankan sekali dan hasilnya sudah ada di repo
ini, jadi notebook dapat langsung dijalankan. Skrip pengunduh sengaja tidak
digabung ke notebook karena tidak termasuk alur praproses. Daymet diambil lewat
server OPeNDAP NASA Earthdata, karena layanan NCSS THREDDS ORNL yang dipakai
`pydaymet` sudah tidak tersedia.

Notebook terbagi delapan bagian: data understanding 23 pos dan uji gap buatan
(Bagian 0, tahap 0a–0j), data mentah debit (1–7), metadata, iklim Daymet, dan
atribut (8–13), hujan stasiun pembanding (14–17), hubungan antarstasiun
(18–19), praproses (20–26), verifikasi dan penyimpanan (27–29), serta persiapan
eksperimen ablasi (30). Seluruh pemeriksaan `assert` ada di satu sel pada tahap
27 dan harus lolos sebelum keluaran ditulis. Notebook dapat dijalankan dari
akar repo maupun dari folder `preprocessing/`. Jalannya sekitar 15 menit,
hampir seluruhnya untuk uji gap buatan di tahap 0i. Koordinat stasiun
hujan dibaca dari `raw/hujan/stasiun_terpakai.txt`, yaitu 16 baris saringan
`ghcnd-stations.txt` yang disimpan di repo, sehingga notebook tidak memerlukan
jaringan dan hasilnya tidak berubah bila NOAA merevisi daftar stasiunnya.

Catatan: `00c` memakai pos terakhir pada daftar sebagai pos target. Urutan
daftarnya tidak penting di tahap unduhan, karena pengurutan hulu ke hilir
dikerjakan notebook memakai jarak alur ke pos target. Pos target 02035000
bukan muara Sungai James; sungai masih berlanjut melewati Richmond. Opsi
`--muara` pada `00a` dan nama kolom `jarak_alur_ke_muara_km` di `raw/` adalah
nama historis untuk pos target ini.

---

## Catatan penting soal API USGS

Layanan NWIS mengembalikan **tiga deret sekaligus** untuk parameter debit:
maksimum harian (`00001`), minimum harian (`00002`), dan rata-rata harian
(`00003`). Permintaan tanpa `statCd` akan mengambil deret pertama, yaitu
**maksimum**, dan rekamannya pun lebih pendek — tanpa pesan galat apa pun.
Parameter `statCd=00003` wajib disertakan.

Hal kedua: rentang rekaman pada metadata (`begin_date` dan `end_date`) **tidak
menjamin kesinambungan**. Sebuah pos dapat tercatat punya rekaman 1950–2024
padahal berlubang bertahun-tahun di tengahnya. Karena itu penyaringan pos di
notebook dikerjakan dari data yang sudah terunduh, bukan dari metadata.

Hal ketiga: USGS tidak menulis baris kosong untuk tanggal tanpa data —
**barisnya memang tidak ada**. Akibatnya `isna()` pada hasil baca mentah selalu
nol dan lubang data tidak terlihat. Lubang baru muncul setelah data direindeks
ke kalender harian penuh.

---

## Uji yang ditunda ke tahap pemodelan

Tiga keputusan preprocessing tidak punya dasar langsung di literatur dan belum
bisa diuji tanpa melatih model. Ketiganya diuji di tahap pemodelan:

| Keputusan | Pilihan sekarang | Uji | Dasar sementara |
|---|---|---|---|
| Panjang window | 14 hari (≈ 1,25 τ, tahap 23) | Bandingkan window 7, 14, 21, dan 28 hari | Aturan ≈ 1,25 τ pilihan sendiri; Kratzert dkk. (2018) juga memilih 365 hari secara heuristik |
| Transformasi hujan | log1p lalu standardisasi (tahap 24 dan 26) | Bandingkan dengan log10(√x + 0,1) dari Feng dkk. (2020) | Prinsip transformasi dari Feng dkk. (2020); bentuk fungsinya pilihan sendiri |
| Urutan stasiun | Jarak alur ke pos target (tahap 12) | Ablasi `acak_stasiun` dan `urutan_luas` yang sudah disiapkan di tahap 30 | Tidak ada di literatur |

## Keterbatasan yang diketahui

**Hujan stasiun hanya pembanding.** Input hujan utama adalah Daymet rata-rata
DAS. Hujan stasiun GHCN-Daily disiapkan sebagai tensor terpisah,
`processed/tensor_hujan_h*.npz`, sebagai pembanding yang lebih dekat ke desain
Le dkk. (2025). Dua hal perlu diingat saat memakainya. Pertama, hujan stasiun di
tensor memakai tanggal file `.dly` tanpa koreksi jam pengamatan, supaya tidak
ada hujan dari hari target di dalam window; akibatnya 11 stasiun COOP yang
dibaca pagi dan 5 stasiun lain bercampur dengan dua pola pencatatan. Koreksi
(COOP dimajukan satu hari) hanya dipakai untuk pemeriksaan akumulasi,
penambalan, dan korelasi antarstasiun. Kedua, satu nilai
Concord (258,8 mm, 22 September 2003) dikosongkan karena dicurigai akumulasi
beberapa hari, sedangkan kasus serupa di Crozier (92,7 mm, 14 Oktober 2017)
belum diubah.

**Nilai isian pada varian pembanding.** Pada V21 dan V18, nilai isian menumpuk
di awal bagian latih, sedangkan bagian uji berisi data asli. Flag membantu
model membedakan keduanya, tetapi perbandingan V21 dan V18 dengan V17 tetap
perlu dibaca dengan keterbatasan ini.

**Sebaran debit tetap menceng setelah standardisasi.** Kemencengannya 3,6,
karena standardisasi hanya menggeser dan menskalakan tanpa mengubah bentuk
sebaran. Akibatnya kerugian kuadrat didominasi hari banjir dan NSE berbias ke
aliran tinggi. Penanggulangannya: metrik dilaporkan terpisah per rentang
aliran, memakai ambang kuartil yang tersimpan di `processed/metadata.json`. Ambang itu
dihitung dari bagian latih saja.

**Sebagian pos berstatus *Non-reference*.** Menurut klasifikasi GAGES-II, hanya
4 dari 16 pos GAGES-II tergolong *Reference*: 02018000, 02027000, 02027500, dan
02028500, seluruhnya anak sungai kecil. Pos target 02035000 berstatus
*Non-reference* dengan 244 bendungan di hulu dan `STOR_NOR_2009` sebesar 28,7
mm tampungan normal (megaliter/km²). Dibagi limpasan rata-rata `RUNAVE7100`
450,5 mm/tahun, nilainya 0,064 tahun atau sekitar 23 hari aliran rata-rata,
sehingga tergolong regulasi rendah (dor < 0,1). DAS seluas 16.193 km² di Amerika
bagian timur memang tidak mungkin bebas gangguan. Yang penting untuk pemodelan,
pembangunan bendungan terbesar terjadi 1960–1980 dan setelah 1990 hanya
bertambah 13 bendungan, sehingga 2000–2024 merupakan satu rezim yang mantap.

**Sumbu stasiun memaksa jaringan berbentuk pohon menjadi barisan.** Pengurutan
menurut jarak alur membuat pos yang bersebelahan bisa berada di cabang yang
berbeda. Contohnya 02031000 (Mechums, hulu Rivanna) terpisah empat posisi dari
02034000 (Rivanna). Urutan relatif Tye dan Piney (selisih 0,4 km) serta
Rockfish dan Mechums (selisih 1,8 km) juga tidak tegas, karena selisihnya seorde
dengan panjang ruas NHDPlus di sekitarnya. Klaim bahwa
CNN memanfaatkan struktur spasial baru bisa dipertahankan setelah ablasi di
tahap 30 notebook dijalankan.

**Kontribusi komponen spasial diperkirakan menurun pada horizon panjang.**
Karena tunda perambatannya hanya 1 hari, informasi pos hulu paling berguna pada
horizon 1 hari. Pada horizon 2 dan 3 hari model harus bersandar pada dinamika
resesi dan simpanan DAS. Dugaan ini perlu diuji dan berlaku sebagai pemeriksaan
kewajaran terhadap model.

---

## Sumber data

| Sumber | Dipakai untuk |
|---|---|
| [USGS NWIS Daily Values](https://waterservices.usgs.gov/) | Debit harian, parameter 00060, statistik 00003 |
| [USGS NWIS Site Service](https://waterservices.usgs.gov/) | Elevasi, luas DAS, koordinat, katalog deret |
| [USGS NLDI](https://api.water.usgs.gov/docs/nldi/) | Topologi jaringan, poligon DAS, geometri alur |
| [GAGES-II (Falcone, 2011)](https://www.sciencebase.gov/catalog/item/631405bbd34e36012efa304a) | Deskripsi 16 pos: status Ref/Non-ref, derajat regulasi, fraksi salju |
| [Daymet V4 R1](https://doi.org/10.3334/ORNLDAAC/2129) lewat OPeNDAP NASA Earthdata | Hujan, Tmax, Tmin harian rata-rata DAS |
| [NOAA GHCN-Daily](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily) | Curah hujan harian, 16 stasiun (pembanding) |
| NHDPlus V2 VAA lewat [HyRiver](https://docs.hyriver.io/) `pynhd` | Orde sungai, jarak alur, kerapatan alur |
| [gridMET](https://www.climatologylab.org/gridmet.html) | Pengisi 31 Desember tahun kabisat Daymet |
| [USGS 3DEP](https://www.usgs.gov/3d-elevation-program) lewat HyRiver `py3dep` | DEM per DAS |

Data USGS dan NOAA berada dalam domain publik.

Falcone, J.A., 2011, *GAGES-II: Geospatial Attributes of Gages for Evaluating
Streamflow*, U.S. Geological Survey.

Menne, M.J., Durre, I., Vose, R.S., Gleason, B.E., Houston, T.G., 2012, *An
overview of the Global Historical Climatology Network-Daily Database*, Journal
of Atmospheric and Oceanic Technology, 29, 897–910.
