# CNN-LSTM + Monte Carlo Dropout untuk Prediksi Debit Air

Prediksi debit air pada jaringan sungai yang saling terhubung, dengan target di
titik hilir dan horizon 1 sampai 3 hari.

Proyek Individu · Teknologi Informasi, DTETI, Universitas Gadjah Mada

---

## Struktur

```
raw/              data mentah hasil unduhan, belum diolah
preprocessing/    skrip pengunduh 00a–00d dan satu notebook preprocessing
processed/        data siap latih
```

Data di `raw/` disimpan dalam format asli server: nama kolom, satuan, kode
kualitas, dan baris komentar header tetap seperti aslinya, dengan satu file per
pos. Penggabungan, interpolasi, dan penyaringan pos dilakukan di notebook.

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
| `preprocessing/00a` – `00d` | Pengunduh, satu per sumber data |
| `preprocessing/preprocessing.ipynb` | Seluruh pengolahan beserta gambar dan tabel, bertahap dalam satu notebook |
| `processed/tensor_h{1,2,3}.npz` | Data siap latih per horizon |
| `processed/tensor_hujan_h{1,2,3}.npz` | Hujan per pos, terpisah, sejajar dengan tensor debit |
| `processed/hujan/` | Hujan harian per stasiun dan per pos, mutu stasiun, `metadata_hujan.json` |
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

Karena USGS tidak menyediakan curah hujan di jaringan ini, data hujan diambil
dari NOAA GHCN-Daily: 16 stasiun di dalam dan sekitar DAS. Tujuh di
antaranya berkelengkapan di atas 99 persen pada data sah mentah, dan delapan
setelah penambalan.

---

## Dataset

**Sungai James, Virginia, Amerika Serikat.** Sumber USGS dan NOAA, domain publik.

| | |
|---|---|
| Pos terpakai | **16** |
| Pos target | 02035000 James River at Cartersville, VA |
| Periode | 1 Januari 2000 – 31 Desember 2024 |
| Jumlah hari | 9.132, kelengkapan 100 persen |
| Nilai hasil interpolasi | 2 dari 146.112 sel (0,0014 persen) |
| Luas DAS target | 16.193 km² |
| Jarak alur terpanjang | 294,1 km |

Enam belas pos, diurutkan dari hulu ke hilir menurut **jarak alur sungai**:

| # | Kode pos | Nama | Elevasi (m) | Luas (km²) | Jarak alur ke pos target (km) |
|---|---|---|---|---|---|
| 1 | 02018000 | Craig Creek at Parr | 302,4 | 852 | 294,1 |
| 2 | 02020500 | Calfpasture River at Goshen | 421,3 | 365 | 281,2 |
| 3 | 02021500 | Maury River at Rockbridge Baths | 335,2 | 852 | 265,7 |
| 4 | 02022500 | Kerrs Creek near Lexington | 298,9 | 91 | 249,4 |
| 5 | 02019500 | James River at Buchanan | 244,5 | 5.369 | 241,4 |
| 6 | 02024000 | Maury River near Buena Vista | 257,8 | 1.676 | 231,8 |
| 7 | 02025500 | James River at Holcomb Rock | 167,0 | 8.433 | 181,2 |
| 8 | 02027000 | Tye River near Lovingston | 175,1 | 241 | 135,9 |
| 9 | 02027500 | Piney River at Piney River | 192,3 | 124 | 132,4 |
| 10 | 02026000 | James River at Bent Creek | 115,9 | 9.451 | 114,3 |
| 11 | 02031000 | Mechums River near White Hall | 130,7 | 247 | 110,2 |
| 12 | 02028500 | Rockfish River near Greenfield | 160,8 | 246 | 109,6 |
| 13 | 02030000 | Hardware River near Scottsville | 89,9 | 300 | 57,4 |
| 14 | 02029000 | James River at Scottsville | 77,0 | 11.865 | 49,9 |
| 15 | 02034000 | Rivanna River at Palmyra | 63,9 | 1.717 | 39,2 |
| 16 | **02035000** | **James River at Cartersville** | 50,9 | 16.193 | 0 (target) |

Tujuh pos lain punya debit harian tetapi tidak terpakai: enam rekamannya baru
dimulai antara 2002 dan 2020, dan satu (02032640) lengkap sejak 2000 tetapi
berlubang 125 hari berturut-turut. Seluruhnya tetap ada di `raw/` dan tercatat
di `processed/mutu_semua_pos.csv`.

---

## Data siap latih

```
X : (contoh, 14 langkah waktu, 16 stasiun, 11 saluran)
y : (contoh,)
```

| Saluran | Isi | Sumber |
|---|---|---|
| 0 | Debit harian, sudah diskalakan | NWIS |
| 1 | Jarak alur ke pos target (km) | NLDI |
| 2 | Kemiringan dasar sungai ke pos target (m/km) | NWIS + NLDI |
| 3 | Elevasi pos (m) | NWIS |
| 4 | Luas DAS, dilogaritmakan | NWIS |
| 5 | Indeks kebulatan DAS | NLDI poligon |
| 6 | Elevasi rata-rata DAS (m) | GAGES-II |
| 7 | Kemiringan rata-rata DAS (persen) | GAGES-II |
| 8 | Curah hujan rata-rata tahunan (cm) | GAGES-II |
| 9 | Indeks aliran dasar | GAGES-II |
| 10 | Kerapatan alur (km/km²) | GAGES-II |

Saluran 1 sampai 10 bersifat statis dan disalin sepanjang sumbu waktu.

Kode bagian: `0` latih, `1` validasi, `2` uji, `3` uji tetapi dikecualikan
karena nilai targetnya hasil perkiraan (`A:e`).

Pengecualian itu hanya berlaku pada nilai yang diprediksi. Jendela masukan
contoh berkode `2` tetap boleh memuat nilai perkiraan: dari 1.322 contoh uji
horizon 1, sebanyak 301 memuat `A:e` pada jendela pos target dan 938 memuat
`A:e` pada pos mana pun. Jadi kode `2` berarti targetnya hasil ukur, bukan
seluruh masukannya hasil ukur.

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
| Jumlah pos | 16 | Lubang maksimum 2 hari, masuk kategori interpolasi linear |
| Satuan | m³/s | Dikonversi dari ft³/s, faktor 0,0283168466 |
| Statistik harian | Rata-rata, `statCd=00003` | Lihat catatan API di bawah |
| Lubang 1–2 hari | Interpolasi linear | Debit berubah lambat |
| Lubang 3–7 hari | Interpolasi, ditandai | Dikeluarkan dari data uji |
| Lubang di atas 7 hari | Pos dibuang | Interpolasinya akan jadi karangan |
| Pembagian | 70 : 15 : 15 menurut urutan waktu | Periode uji tidak disentuh sampai akhir |
| Penskalaan | Dihitung dari bagian latih saja, `ddof=0` | Mencegah kebocoran informasi masa depan |
| Transformasi debit | Standardisasi saja | Lihat keterbatasan di bawah |
| Urutan sumbu stasiun | Menurut jarak alur sungai ke pos target | Menyatakan posisi di jaringan |
| Panjang jendela | 14 hari | 1,25 × konstanta waktu resesi, lihat di bawah |
| Horizon | 1, 2, 3 hari | Dilatih terpisah tiap horizon |
| Masukan runtun waktu | Debit saja | Hujan jadi eksperimen terpisah |
| Atribut statis | 10 dari 571 | Dipilih menurut relevansi fisik |
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
sebelumnya; tidak ada kejadian resesi yang dipilih secara khusus.

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

Indeks aliran dasar (`BFI_AVE`, saluran 9) juga diturunkan dari rekaman debit,
sama seperti `FlowRec`, tetapi tetap dipakai. Perbedaannya, `BFI_AVE`
menggambarkan sifat perilaku DAS, yaitu porsi aliran dari simpanan air tanah,
dan lazim dipakai sebagai deskriptor DAS pada literatur hidrologi termasuk
kumpulan data CAMELS, sedangkan `FlowRec` hanya menggambarkan riwayat
pemantauan.

---

## Menghasilkan ulang

Butuh `pandas`, `numpy`, `matplotlib`, `requests`, `openpyxl`, `jupyter`. Tanpa
akun, tanpa kunci API.

```bash
# 1. debit + jaringan + metadata  ->  raw/{network,sites,discharge}/
python3 preprocessing/00a_unduh_debit_dan_jaringan.py \
    --muara 02035000 --jarak 300 --mulai 2000-01-01 --selesai 2024-12-31 \
    --output raw

POS="02018000 02019500 02020500 02021500 02022500 02024000 02025500 02026000
     02027000 02027500 02028500 02029000 02030000 02031000 02034000 02035000"

# 2. atribut DAS GAGES-II  ->  raw/gagesii/
python3 preprocessing/00b_unduh_gagesii.py $POS --output raw/gagesii

# 3. atribut spasial turunan  ->  raw/spatial/
python3 preprocessing/00c_atribut_spasial.py $POS -o raw/spatial/atribut_turunan.csv

# 4. curah hujan GHCN-Daily  ->  raw/hujan/
python3 preprocessing/00d_unduh_hujan.py raw/hujan

# 5. preprocessing  ->  processed/
jupyter nbconvert --to notebook --execute --inplace preprocessing/preprocessing.ipynb
```

Langkah 1 sampai 4 hanya perlu dijalankan sekali dan hasilnya sudah ada di repo
ini, jadi notebook dapat langsung dijalankan. Skrip pengunduh sengaja tidak
digabung ke notebook karena tidak termasuk alur praproses.

Notebook terbagi tujuh bagian dan 30 tahap: data mentah debit (1–7), metadata
dan atribut (8–13), curah hujan (14–17), hubungan antarstasiun (18–19),
praproses (20–26), verifikasi dan penyimpanan (27–29), serta persiapan
eksperimen ablasi (30). Seluruh
pemeriksaan `assert` ada di satu sel pada tahap 27 dan harus lolos sebelum
keluaran ditulis. Notebook dapat dijalankan dari akar repo maupun dari folder
`preprocessing/`; jalannya sekitar setengah menit. Koordinat stasiun
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

## Keterbatasan yang diketahui

**Curah hujan harian belum dipakai di model dasar.** Baseline memakai debit
saja, mengikuti Ghimire dkk. (2021). Hujan sudah disiapkan sebagai tensor
terpisah, `processed/tensor_hujan_h*.npz`, untuk eksperimen lanjutan. Dua hal
perlu diingat saat memakainya. Pertama, 11 stasiun COOP yang dibaca pagi hari
dimajukan satu hari agar sekalender dengan debit; bacaan pagi itu memuat
sekitar 7 jam pertama hari berikutnya, sehingga hasil eksperimen sebaiknya
dibandingkan dengan data hujan tanpa koreksi. Kedua, satu nilai Concord
(258,8 mm, 22 September 2003) dikosongkan karena dicurigai akumulasi beberapa
hari, sedangkan kasus serupa di Crozier (92,7 mm, 14 Oktober 2017) belum
diubah.

**Sebaran debit tetap menceng setelah standardisasi.** Kemencengannya 3,6,
karena standardisasi hanya menggeser dan menskalakan tanpa mengubah bentuk
sebaran. Akibatnya kerugian kuadrat didominasi hari banjir dan NSE berbias ke
aliran tinggi. Penanggulangannya: metrik dilaporkan terpisah per rentang
aliran, memakai ambang kuartil yang tersimpan di `processed/metadata.json`. Ambang itu
dihitung dari bagian latih saja.

**Sebagian pos berstatus *Non-reference*.** Menurut klasifikasi GAGES-II, hanya
4 dari 16 pos tergolong *Reference*: 02018000, 02027000, 02027500, dan
02028500, seluruhnya anak sungai kecil. Pos target 02035000 berstatus
*Non-reference* dengan 244 bendungan di hulu dan `STOR_NOR_2009` sebesar 28,7
hari aliran rata-rata — tergolong sedang. DAS seluas 16.193 km² di Amerika
bagian timur memang tidak mungkin bebas gangguan. Yang penting untuk pemodelan,
pembangunan bendungan terbesar terjadi 1960–1980 dan setelah 1990 hanya
bertambah 13 bendungan, sehingga 2000–2024 merupakan satu rezim yang mantap.

**Sumbu stasiun memaksa jaringan berbentuk pohon menjadi barisan.** Pengurutan
menurut jarak alur membuat pos yang bersebelahan bisa berada di cabang yang
berbeda. Contohnya 02031000 (Mechums, hulu Rivanna) terpisah tiga posisi dari
02034000 (Rivanna). Urutan relatif 02031000 (110,2 km) dan 02028500 (109,6 km)
juga tidak tegas, karena selisih 0,6 km lebih kecil daripada panjang segmen
alur NHDPlus rata-rata di lintasan keduanya (1,0 sampai 1,6 km). Klaim bahwa
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
| [GAGES-II (Falcone, 2011)](https://www.sciencebase.gov/catalog/item/631405bbd34e36012efa304a) | 571 atribut DAS per pos |
| [NOAA GHCN-Daily](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily) | Curah hujan harian, 16 stasiun |

Data USGS dan NOAA berada dalam domain publik.

Falcone, J.A., 2011, *GAGES-II: Geospatial Attributes of Gages for Evaluating
Streamflow*, U.S. Geological Survey.

Menne, M.J., Durre, I., Vose, R.S., Gleason, B.E., Houston, T.G., 2012, *An
overview of the Global Historical Climatology Network-Daily Database*, Journal
of Atmospheric and Oceanic Technology, 29, 897–910.
