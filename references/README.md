# Daftar referensi preprocessing dan pemodelan

Folder ini hanya berisi daftar. **PDF tidak disimpan di repo** karena sebagian
berlangganan dan repo ini publik. Unduh PDF satu per satu lewat SSO UGM (alat
unduh massal dilarang) atau dari tautan gratis yang tercantum, lalu simpan di
komputer sendiri.

Kunci di kolom pertama dipakai di notebook dengan format `Dasar: [Kunci]`.

Status: **V** = isi dan DOI sudah dicek, **V\*** = isi dicek, DOI belum dicek
ulang, **V\*\*** = isi dicek, tidak punya DOI Crossref (lihat bagian G),
**D** = khusus dataset/perangkat lunak.

## A. Data hilang dan pengisian gap

| Kunci | Referensi | DOI / ID | Status | Akses | Dipakai untuk |
|---|---|---|---|---|---|
| Hirsch1982 | Hirsch, R. M. (1982). A comparison of four streamflow record extension techniques. *Water Resources Research*, 18(4), 1081–1088. | 10.1029/WR018i004p01081 | V\* | Wiley (UGM); PDF gratis di ESF | Pemilihan donor berdasarkan korelasi; MOVE.1 |
| ArchfieldVogel2010 | Archfield, S. A., & Vogel, R. M. (2010). Map correlation method: Selection of a reference streamgage to estimate daily streamflow at ungaged catchments. *Water Resources Research*, 46, W10513. | 10.1029/2009WR008481 | V | Wiley (UGM) | Donor bukan pos terdekat |
| Tencaliec2015 | Tencaliec, P., Favre, A.-C., Prieur, C., & Mathevet, T. (2015). Reconstruction of missing daily streamflow data using dynamic regression models. *Water Resources Research*, 51, 9447–9463. | 10.1002/2015WR017399 | V | Wiley (UGM); gratis di HAL | Gap interior panjang (125 hari); desain uji gap buatan |
| Hamzah2020 | Hamzah, F. B., dkk. (2020). Imputation methods for recovering streamflow observation: A methodological review. *Cogent Environmental Science*, 6(1), 1745133. | 10.1080/23311843.2020.1745133 | V | T&F, open access | Tinjauan metode; tidak ada batas panjang gap |
| Asquith2006 | Asquith, W. H., Roussel, M. C., & Vrabel, J. (2006). Statewide analysis of the drainage-area ratio method for 34 streamflow percentile ranges in Texas. USGS SIR 2006-5286. | 10.3133/sir20065286 | V | Gratis (USGS) | Metode rasio luas (DAR), eksponen φ. Catatan: φ ≈ 0,89 berlaku untuk persentil debit 0–50; untuk debit yang lebih tinggi φ-nya 0,92–0,93. Di notebook DAR dengan φ tetap hanya metode pembanding (tahap 0i) |
| Emerson2005 | Emerson, D. G., Vecchia, A. V., & Dahl, A. L. (2005). Evaluation of drainage-area ratio method used to estimate streamflow for the Red River of the North Basin. USGS SIR 2005-5017. | 10.3133/sir20055017 | V (abstrak) | Gratis (USGS) | Pembanding φ musiman |
| Farmer2016 | Farmer, W. H. (2016). Ordinary kriging as a tool to estimate historical daily streamflow records. *Hydrology and Earth System Sciences*, 20, 2721–2735. | 10.5194/hess-20-2721-2016 | V | Gratis (HESS) | Akurasi transfer regional; jaringan bersarang |
| Amisigo2005 | Amisigo, B. A., & van de Giesen, N. C. (2005). Using a spatio-temporal dynamic state-space model with the EM algorithm to patch gaps in daily riverflow series. *HESS*, 9, 209–224. | 10.5194/hess-9-209-2005 | V | Gratis (HESS) | Gap ≤30 hari, informasi target + tetangga |
| Baddoo2021 | Baddoo, T. D., dkk. (2021). Comparison of missing data infilling mechanisms for recovering a real-world single station streamflow observation. *IJERPH*, 18(16), 8375. | 10.3390/ijerph18168375 | V | Gratis (MDPI) | Pembanding metode satu stasiun |
| Nearing2022 | Nearing, G., Klotz, D., dkk. (2022). Technical note: Data assimilation and autoregression for using near-real-time streamflow observations in LSTM networks. *HESS*, 26, 5493–5513. | 10.5194/hess-26-5493-2022 | V | Gratis (HESS) | Flag biner untuk debit lag yang hilang |
| Nearing2024 | Nearing, G., dkk. (2024). Global prediction of extreme floods in ungauged watersheds. *Nature*, 627, 559–563. | 10.1038/s41586-024-07145-1 | V | Open access | Isi dari sumber lain atau rata-rata + flag biner |
| Gauch2025 | Gauch, M., dkk. (2025). How to deal with missing input data. *HESS*, 29, 6221–. | 10.5194/hess-29-6221-2025 | V | Gratis (HESS) | Perbandingan strategi input hilang |
| Che2018 | Che, Z., Purushotham, S., Cho, K., Sontag, D., & Liu, Y. (2018). Recurrent neural networks for multivariate time series with missing values. *Scientific Reports*, 8, 6085. | 10.1038/s41598-018-24271-9 | V\* | Open access | Mengapa imputasi harus diberi penanda |
| Mital2020 | Mital, U., dkk. (2020). Sequential imputation of missing spatio-temporal precipitation data using random forests. *Frontiers in Water*, 2, 20. | 10.3389/frwa.2020.00020 | V | Gratis | Gap hujan diisi dari stasiun tetangga, bukan interpolasi waktu |

## B. Pemilihan variabel dinamis

| Kunci | Referensi | DOI / ID | Status | Akses | Dipakai untuk |
|---|---|---|---|---|---|
| Thornton2021 | Thornton, P. E., Shrestha, R., Thornton, M., Kao, S.-C., Wei, Y., & Wilson, B. E. (2021). Gridded daily weather data for North America with comprehensive uncertainty quantification. *Scientific Data*, 8, 190. | 10.1038/s41597-021-00973-0 | V | Open access (Europe PMC PMC8302764) | Definisi hari Daymet V4 (tengah malam ke tengah malam) dan koreksi jam pengamatan stasiun |
| Bardossy2008 | Bárdossy, A., & Das, T. (2008). Influence of rainfall observation network on model calibration and application. *Hydrology and Earth System Sciences*, 12(1), 77–89. | 10.5194/hess-12-77-2008 | DOI dicek di Crossref; isi belum dicek | Open access (HESS) | Pengaruh kerapatan jaringan stasiun hujan; dasar perapatan jaringan v2 (preprocessing tahap 31) |
| Kratzert2018 | Kratzert, F., dkk. (2018). Rainfall–runoff modelling using Long Short-Term Memory (LSTM) networks. *HESS*, 22, 6005–6022. | 10.5194/hess-22-6005-2018 | V | Gratis | Set forcing; SWE tidak dipakai; target tanpa transformasi log, cukup standardisasi; statistik scaling dari periode kalibrasi saja; panjang input 365 hari dipilih heuristik |
| Kratzert2019 | Kratzert, F., dkk. (2019). Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets. *HESS*, 23, 5089–5110. | 10.5194/hess-23-5089-2019 | V | Gratis | Forcing Daymet; atribut statis dan sensitivitasnya; atribut statis "concatenated to the meteorological inputs at each time step" |
| Kratzert2021 | Kratzert, F., dkk. (2021). A note on leveraging synergy in multiple meteorological data sets with deep learning for rainfall–runoff modeling. *HESS*, 25, 2685–2703. | 10.5194/hess-25-2685-2021 | V | Gratis | Presipitasi sebagai variabel terpenting |
| Feng2020 | Feng, D., Fang, K., & Shen, C. (2020). Enhancing streamflow forecast and extracting insights using LSTM networks with data integration at continental scales. *Water Resources Research*, 56, e2019WR026793. | 10.1029/2019WR026793 | V | Wiley; gratis arXiv 1912.08949 | Debit lag menaikkan NSE; prinsip mentransformasi hujan berdistribusi Gamma, log10(√x + 0,1) |
| Lees2021 | Lees, T., dkk. (2021). Benchmarking data-driven rainfall–runoff models in Great Britain. *HESS*, 25, 5517–5534. | 10.5194/hess-25-5517-2021 | V | Gratis | Forcing P, T, PET; `reservoir_cap` |
| Lees2022 | Lees, T., dkk. (2022). Hydrological concept formation inside LSTM networks. *HESS*, 26, 3079–3101. | 10.5194/hess-26-3079-2022 | V | Gratis | Salju dan kelembapan tanah tidak perlu jadi input |
| Choi2022 | Choi, J., dkk. (2022). Learning enhancement method of LSTM network and its applicability in hydrological time series prediction. *Water*, 14(18), 2910. | 10.3390/w14182910 | V | Gratis (MDPI) | Tambahan forcing tidak menaikkan median NSE |
| GauchMaiLin2021 | Gauch, M., Mai, J., & Lin, J. (2021). The proper care and feeding of CAMELS: How limited training data affects streamflow prediction. *Environmental Modelling & Software*, 135, 104926. | 10.1016/j.envsoft.2020.104926 | V | Elsevier (tidak dilanggan); gratis arXiv 1911.07249 | Periode latih tidak boleh dipendekkan |
| Stein2021 | Stein, L., dkk. (2021). How do climate and catchment attributes influence flood generating processes? *Water Resources Research*, 57, e2020WR028300. | 10.1029/2020WR028300 | V | Wiley; PDF penulis gratis | Ambang DAS salju 20% |
| Berghuijs2014 | Berghuijs, W. R., Woods, R. A., & Hrachowitz, M. (2014). A precipitation shift from snow towards rain leads to a decrease in streamflow. *Nature Climate Change*, 4, 583–586. | 10.1038/nclimate2246 | V | AAM gratis (core.ac.uk) | Ambang salju 15% |
| Ouyang2021 | Ouyang, W., dkk. (2021). Continental-scale streamflow modeling of basins with reservoirs. *Journal of Hydrology*, 599, 126455. | 10.1016/j.jhydrol.2021.126455 | V | Elsevier (tidak dilanggan); gratis arXiv 2101.04423 | Degree of regulation; kelas small-dor |
| Steyaert2022 | Steyaert, J. C., Condon, L. E., Turner, S. W. D., & Voisin, N. (2022). ResOpsUS, a dataset of historical reservoir operations in the contiguous United States. *Scientific Data*, 9, 34. | 10.1038/s41597-022-01134-7 | V | Open access | Data operasi waduk di AS timur terbatas |
| Kennedy1984 | Kennedy, E. J. (1984). Discharge ratings at gaging stations. USGS TWRI 3-A10. | 10.3133/twri03A10 | V | Gratis (USGS) | Tinggi muka air redundan dengan debit |
| FentonKeller2001 | Fenton, J. D., & Keller, R. J. (2001). The calculation of streamflow from measurements of stage. CRC for Catchment Hydrology, Technical Report 01/6. | — | V | Gratis | Efek hysteresis kecil |

## C. Atribut statis

| Kunci | Referensi | DOI / ID | Status | Akses | Dipakai untuk |
|---|---|---|---|---|---|
| Addor2017 | Addor, N., Newman, A. J., Mizukami, N., & Clark, M. P. (2017). The CAMELS data set: catchment attributes and meteorology for large-sample studies. *HESS*, 21, 5293–5313. | 10.5194/hess-21-5293-2017 | V | Gratis | Definisi atribut; BFI adalah signature debit |
| Li2022 | Li, X., dkk. (2022). Regionalization in a global hydrologic deep learning model: From physical descriptors to random vectors. *Water Resources Research*, 58(8), e2021WR031794. | 10.1029/2021WR031794 | V\* (judul/jurnal; DOI dicek lewat Crossref) | Wiley; versi NSF PAR | Kontrol atribut acak |
| Hashemi2022 | Hashemi, R., Brigode, P., Garambois, P.-A., & Javelle, P. (2022). How can we benefit from regime information to make more effective use of LSTM runoff models? *HESS*, 26, 5793–5816. | 10.5194/hess-26-5793-2022 | V | Gratis | Signature dari periode latih saja |
| Ladson2013 | Ladson, A. R., Brown, R., Neal, B., & Nathan, R. (2013). A standard approach to baseflow separation using the Lyne and Hollick filter. *Australasian Journal of Water Resources*, 17(1), 25–34. | 10.7158/13241583.2013.11465417 | V | T&F | Filter Lyne–Hollick untuk BFI bagian latih (α = 0,925, 3 lintasan, pantulan 30 hari) |
| Bassi2024 | Bassi, A., dkk. (2024). Learning landscape features from streamflow with autoencoders. *HESS*, 28, 4971–. | 10.5194/hess-28-4971-2024 | V | Gratis | Atribut statis sebagai identitas DAS |
| Falcone2011 | Falcone, J. A. (2011). GAGES-II: Geospatial attributes of gages for evaluating streamflow. USGS. | 10.3133/70046617 | V | Gratis (USGS) | Sumber atribut 16 pos; STOR_NOR_2009, RUNAVE7100 |

## D. Dataset dan layanan data

| Kunci | Referensi | DOI / ID | Status | Dipakai untuk |
|---|---|---|---|---|
| NWIS | U.S. Geological Survey. National Water Information System, debit harian parameter 00060 statistik 00003. | waterservices.usgs.gov | D | Debit 23 pos |
| NLDI | U.S. Geological Survey. Network-Linked Data Index (navigasi DM/UT/UM, poligon DAS). | api.water.usgs.gov/nldi | D | Topologi jaringan, poligon DAS, peta |
| Daymet | Thornton, M. M., dkk. Daymet: Daily Surface Weather Data on a 1-km Grid for North America, Version 4 R1. ORNL DAAC. | 10.3334/ORNLDAAC/2129 | D | P, Tmax, Tmin rata-rata DAS |
| gridMET | Abatzoglou, J. T. (2013). Development of gridded surface meteorological data for ecological applications and modelling. *Int. J. Climatology*, 33, 121–131. | 10.1002/joc.3413 (cek ulang) | D | Pengisi 31 Desember tahun kabisat Daymet |
| GHCND | Menne, M. J., dkk. Global Historical Climatology Network – Daily, Version 3. NOAA NCEI. | 10.7289/V5D21VHZ (cek ulang) | D | Hujan stasiun (pembanding) |
| HPD | NOAA NCEI. COOP Hourly Precipitation Data (C-HPD), Version 2.0; dokumentasi di ncei.noaa.gov/data/coop-hourly-precipitation/v2/doc/ (readme.csv.txt, CHPD-v2-ATBD-20181023.pdf). | — | D | Hujan per jam Gathright Dam, Piedmont, Mill Gap (preprocessing tahap 33) |
| NHDPlusV2 | McKay, L., Bondelid, T., Dewald, T., Johnston, J., Moore, R., & Rea, A. (2012). NHDPlus Version 2: User Guide. U.S. EPA. | — | D | Orde Strahler (VAA) alur di dalam DAS; alur orde ≥ 3 di luar DAS sebagai latar peta |
| 3DEP | U.S. Geological Survey. 3D Elevation Program (DEM 1 arc-second, diambil lewat py3dep). | — | D | Elevasi dan kemiringan DAS; relief latar peta |
| HyRiver | Chegini, T., Li, H.-Y., & Leung, L. R. (2021). HyRiver: Hydroclimate data retriever. *Journal of Open Source Software*, 6(66), 3175. | 10.21105/joss.03175 (cek ulang) | D | Pustaka unduh Daymet/NLDI/3DEP |
| Sklearn | Pedregosa, F., dkk. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830; ditambah dokumentasi `StandardScaler`. | — | D | `ddof = 0` pada standardisasi (dokumentasi `StandardScaler` memakai standar deviasi bias) |

## E. Paper acuan arsitektur

| Kunci | Referensi | DOI / ID | Status | Akses | Dipakai untuk |
|---|---|---|---|---|---|
| Le2025 | Le, X.-H., Binh, D. V., & Lee, G. (2025). Performance and uncertainty analysis in deep learning frameworks for streamflow forecasting via Monte Carlo dropout technique. *Journal of Hydrology: Regional Studies*, 61, 102668. | 10.1016/j.ejrh.2025.102668 | V | Open access | Acuan tensor, matriks korelasi, peta pos; pembagian kronologis per tahun (hlm. 5); horizon t+1 s.d. t+4 dengan target terpisah (hlm. 4–5); Tabel 2 |

## F. Pembagian data, evaluasi, resesi, dan interpolasi

| Kunci | Referensi | DOI / ID | Status | Akses | Dipakai untuk |
|---|---|---|---|---|---|
| Roberts2017 | Roberts, D. R., dkk. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40, 913–929. | 10.1111/ecog.02881 | V\* | Wiley | Pembagian menurut waktu pada data berautokorelasi |
| Krause2005 | Krause, P., Boyle, D. P., & Bäse, F. (2005). Comparison of different efficiency criteria for hydrological model assessment. *Advances in Geosciences*, 5, 89–97. | 10.5194/adgeo-5-89-2005 | V | Gratis | Dominasi debit tinggi pada NSE dan MSE |
| McMillan2012 | McMillan, H., Krueger, T., & Freer, J. (2012). Benchmarking observational uncertainties for hydrology: rainfall, river discharge and water quality. *Hydrological Processes*, 26(26), 4078–4111. | 10.1002/hyp.9384 | V | Wiley | Ketidakpastian debit bergantung pada kondisi aliran dan metode pengukuran |
| Shepard1968 | Shepard, D. (1968). A two-dimensional interpolation function for irregularly-spaced data. *Proceedings of the 1968 ACM National Conference*, 517–524. | 10.1145/800186.810616 | V | ACM | IDW pangkat 2 |
| Stoelzle2013 | Stoelzle, M., Stahl, K., & Weiler, M. (2013). Are streamflow recession characteristics really characteristic? *HESS*, 17, 817–828. | 10.5194/hess-17-817-2013 | V | Gratis | Karakteristik resesi bergantung pada metode ekstraksi |
| Tallaksen1995 | Tallaksen, L. M. (1995). A review of baseflow recession analysis. *Journal of Hydrology*, 165, 349–370. | 10.1016/0022-1694(94)02540-R | DOI cek ulang | Elsevier (tidak dilanggan) | Model resesi eksponensial |

## G. Pemodelan: arsitektur, pelatihan, pemilihan, dan ketidakpastian

DOI dicek ke Crossref pada 19 November 2026 (judul, penulis, wadah, volume,
halaman cocok). Tiga entri tanpa DOI Crossref diberi status **V\*\***: isi
dicek, ID arXiv dicek di DataCite atau tidak ada DOI sama sekali.

| Kunci | Referensi | DOI / ID | Status | Akses | Dipakai untuk |
|---|---|---|---|---|---|
| Kingma2014 | Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv:1412.6980. | arXiv:1412.6980 (10.48550/arXiv.1412.6980, DataCite) | V\*\* | Gratis (arXiv) | Optimizer Adam; learning rate bawaan 0,001 |
| He2015 | He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. *Proc. IEEE ICCV*, 1026–1034. | 10.1109/ICCV.2015.123 | V | IEEE; gratis di CVF Open Access | Inisialisasi he_normal |
| Prechelt1998 | Prechelt, L. (1998). Early stopping — but when? Dalam G. B. Orr & K.-R. Müller (Ed.), *Neural Networks: Tricks of the Trade*, LNCS 1524, 55–69. Springer. | 10.1007/3-540-49430-8_3 | V | Springer | Early stopping pada galat validasi |
| Gupta2009 | Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009). Decomposition of the mean squared error and NSE performance criteria: Implications for improving hydrological modelling. *Journal of Hydrology*, 377(1–2), 80–91. | 10.1016/j.jhydrol.2009.08.003 | V | Elsevier | KGE |
| Ghobadi2022 | Ghobadi, F., & Kang, D. (2022). Multi-step ahead probabilistic forecasting of daily streamflow using Bayesian deep learning: A multiple case study. *Water*, 14(22), 3672. | 10.3390/w14223672 | V | Gratis (MDPI) | PICP dan MPIW pada interval 95% |
| Ghimire2021 | Ghimire, S., Yaseen, Z. M., Farooque, A. A., Deo, R. C., Zhang, J., & Tao, X. (2021). Streamflow prediction using an integrated methodology based on convolutional neural network and long short-term memory networks. *Scientific Reports*, 11, 17497. | 10.1038/s41598-021-96751-4 | V | Open access | CNN-LSTM: Adam lr 0,001, LSTM 100 unit, dropout 0,1, batch 500, 300 epoch, patience 20; 3 lapisan Conv1D (80, 50, 10 filter) (Tabel 2–3) |
| Pokharel2024 | Pokharel, S., & Roy, T. (2024). A parsimonious setup for streamflow forecasting using CNN-LSTM. *Journal of Hydroinformatics*, 26(11), 2751–2761. Pracetak arXiv:2404.07924. | 10.2166/hydro.2024.114 | V | IWA; gratis arXiv | CNN-LSTM: LSTM 80 unit, dropout 0,3, batch 50, 100 epoch, He uniform; 3 lapisan konvolusi (32, 16, 32), kernel 1×1 dan 3×3. Catatan: halaman akhir 2761 (Crossref dan PDF), bukan 2776 |
| Weißenborn2025 | Weißenborn, M., Breuer, L., & Houska, T. (2025). Neural networks in catchment hydrology: a comparative study of different algorithms in an ensemble of ungauged basins in Germany. *HESS*, 29, 5131–5164. | 10.5194/hess-29-5131-2025 | V | Gratis (HESS) | Pencarian Latin hypercube (100 sampel); batch 256 lebih baik daripada 2048; jendela 50–365 hari |
| Klotz2022 | Klotz, D., Kratzert, F., Gauch, M., Keefe Sampson, A., Brandstetter, J., Klambauer, G., Hochreiter, S., & Nearing, G. (2022). Uncertainty estimation with deep learning for rainfall–runoff modeling. *HESS*, 26(6), 1673–1693. | 10.5194/hess-26-1673-2022 | V | Gratis (HESS) | Model MC Dropout (Tabel A3): dropout 0,75 dari kandidat 0,1–0,75, 500 sel LSTM, learning rate 0,001, batch 256. Hiperparameter dipilih dari kalibrasi, yaitu simpangan terkecil dari garis 1:1 probability plot, **dihitung pada data latih** (bagian metode, "selected on the basis of the training data"). MC Dropout memberi estimasi ketidakpastian paling buruk di antara empat pendekatan (kesimpulan); 7.500 sampel |
| Gal2016 | Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. *Proceedings of the 33rd International Conference on Machine Learning*, PMLR 48, 1050–1059. | arXiv:1506.02142 (10.48550/arXiv.1506.02142, DataCite); PMLR tanpa DOI | V\*\* | Gratis (PMLR) | MC Dropout: dropout tetap aktif saat prediksi |
| Srivastava2014 | Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research*, 15, 1929–1958. | — (JMLR tanpa DOI) | V\*\* | Gratis (JMLR) | Dropout |
| Hochreiter1997 | Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780. | 10.1162/neco.1997.9.8.1735 | V | MIT Press | LSTM |
| LeCun1998 | LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324. | 10.1109/5.726791 | V | IEEE | Konvolusi dengan bobot bersama |
| Donahue2017 | Donahue, J., Hendricks, L. A., Rohrbach, M., Venugopalan, S., Guadarrama, S., Saenko, K., & Darrell, T. (2017). Long-term recurrent convolutional networks for visual recognition and description. *IEEE TPAMI*, 39(4), 677–691. | 10.1109/TPAMI.2016.2599174 | V | IEEE; gratis arXiv 1411.4389 | CNN per langkah waktu, lalu LSTM atas urutan vektor fitur |

## Jangan dikutip dulu

Isinya belum terbaca atau statusnya belum jelas: Harvey dkk. 2012, Gyau-Boakye &
Schultz 1994, Vogel & Stedinger 1985 (MOVE.3), Kirschstein & Sun 2024, Sit 2021,
Strahler/Horton/Miller/Schumm untuk rasio kebulatan. Pracetak Weißenborn sudah
digantikan versi terbitnya (Weißenborn2025, bagian G).
