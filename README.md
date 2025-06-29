# Proyek Analisis Prediktif: Prediksi Tingkat Polusi Udara

- **Nama:** Rahmat Amalul Ahlin
- **Email:** rahmatmulyan@gmail.com
- **ID Dicoding:** rahmat_amalul

![image-7](https://github.com/user-attachments/assets/d8c02a44-82d1-4bde-9d70-5c0143444c79)  

## Domain Proyek
Polusi udara terus menjadi salah satu ancaman kesehatan lingkungan terbesar di dunia, terutama di kota-kota besar seperti Jakarta. Laporan terbaru secara konsisten menempatkan Jakarta sebagai salah satu kota dengan kualitas udara terburuk di dunia. Menurut data dari platform pemantau kualitas udara IQAir, tingkat konsentrasi PM2.5 di Jakarta seringkali melampaui ambang batas aman yang ditetapkan oleh Organisasi Kesehatan Dunia (WHO) [[1]](https://www.iqair.com/world-most-polluted-cities). Situasi yang mengkhawatirkan ini diperparah oleh faktor-faktor seperti emisi dari sektor transportasi yang padat, aktivitas industri, dan pembakaran sampah, yang secara kolektif meningkatkan risiko kesehatan bagi jutaan penduduknya.

Partikel PM2.5, yang berukuran lebih kecil dari 2.5 mikrometer, menjadi perhatian utama karena kemampuannya untuk menembus jauh ke dalam sistem pernapasan dan bahkan masuk ke aliran darah. Paparan jangka panjang terhadap PM2.5 terbukti secara ilmiah memiliki kaitan erat dengan peningkatan risiko penyakit serius, termasuk infeksi saluran pernapasan akut (ISPA), penyakit jantung iskemik, stroke, dan kanker paru-paru [[2]](https://pubs.acs.org/doi/10.1021/acs.estlett.8b00360). Mengingat dampak kesehatan yang signifikan dan kerugian ekonomi yang ditimbulkannya, penanganan masalah polusi udara menjadi sangat mendesak.

Salah satu pendekatan proaktif untuk mitigasi dampak polusi adalah dengan mengembangkan sistem prediksi yang akurat. Dengan memprediksi kapan tingkat polusi akan mencapai level berbahaya, pemerintah dapat mengeluarkan peringatan dini kepada masyarakat, terutama kelompok rentan, dan menerapkan kebijakan intervensi sementara seperti pembatasan lalu lintas. Riset sebelumnya telah menunjukkan bahwa model machine learning sangat efektif dalam memprediksi konsentrasi PM2.5 dengan memanfaatkan data historis dari polutan lain dan variabel meteorologi [[3]](https://www.research.herts.ac.uk/ws/files/32433439/conference_paper_2_Ismail_etal.pdf), [[4]](https://link.springer.com/article/10.1007/s11356-020-09855-1). Proyek ini bertujuan untuk menerapkan pendekatan serupa guna membangun model prediksi yang kuat dan andal sebagai alat bantu pengambilan keputusan untuk kualitas udara yang lebih baik.

## Business Understanding

### Problem Statement
- Tingkat polusi udara PM2.5 yang tinggi di kota-kota besar merupakan ancaman serius bagi kesehatan publik. Namun, masyarakat dan lembaga pemerintah seringkali tidak memiliki alat yang andal untuk mengantisipasi hari-hari dengan kualitas udara yang buruk, sehingga sulit untuk mengambil tindakan pencegahan.  
- Meskipun diketahui bahwa kondisi cuaca dan polutan lain memengaruhi PM2.5, seringkali tidak jelas faktor spesifik mana yang menjadi pendorong utama di suatu lokasi. Tanpa pemahaman ini, upaya intervensi kebijakan (seperti pembatasan lalu lintas) bisa menjadi tidak efisien.  

### Goals
- Membangun sebuah model machine learning yang mampu memprediksi konsentrasi PM2.5 dengan akurat untuk beberapa waktu ke depan berdasarkan data historis polusi dan meteorologi.  
- Menganalisis model yang telah dikembangkan untuk mengidentifikasi dan memeringkat fitur-fitur (cuaca, polutan lain) yang paling berpengaruh terhadap fluktuasi tingkat PM2.5.  

### Solution Statements  
- Untuk mendapatkan model prediksi yang akurat, saya akan membangun dan mengevaluasi beberapa algoritma regresi yang berbeda: Random Forest Regressor, Support Vector Regressor (SVR), dan K-Neighbors Regressor (KNN). Model dengan performa terbaik kemudian akan dioptimalkan lebih lanjut melalui proses hyperparameter tuning untuk meningkatkan akurasinya.  
- Keberhasilan solusi akan diukur secara kuantitatif menggunakan metrik evaluasi regresi. Metrik utama adalah Mean Absolute Error (MAE) untuk mengetahui rata-rata kesalahan prediksi dan Mean Squared Error (MSE) untuk melihat dampak dari kesalahan besar. Model akan dianggap berhasil jika mampu mencapai nilai MAE dan MSE serendah mungkin pada data uji.  

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah "Beijing PM2.5 Data" yang berfokus pada data dari stasiun pemantauan Aotizhongxin selama periode 2013 hingga 2017. Dataset ini merupakan kumpulan data time-series per jam yang mencatat konsentrasi polutan udara serta data meteorologi. Data ini bersumber dari UCI Machine Learning Repository dan dapat diunduh melalui Kaggle.

- [Link Dataset](https://www.kaggle.com/datasets/shaviranurfadhilla/prsa-data-aotizhongxin-2013-2017)

#### Jumlah data  
|    | Column   | Non-Null Count | Dtype    |
|-----|----------|----------------|----------|
| 0   | No       | 10000          | int64    |
| 1   | year     | 10000          | int64    |
| 2   | month    | 10000          | int64    |
| 3   | day      | 10000          | int64    |
| 4   | hour     | 10000          | int64    |
| 5   | PM2.5    | 9725           | float64  |
| 6   | PM10     | 9787           | float64  |
| 7   | SO2      | 9733           | float64  |
| 8   | NO2      | 9692           | float64  |
| 9   | CO       | 9479           | float64  |
| 10  | O3       | 9527           | float64  |
| 11  | TEMP     | 9992           | float64  |
| 12  | PRES     | 9992           | float64  |
| 13  | DEWP     | 9992           | float64  |
| 14  | RAIN     | 9992           | float64  |
| 15  | wd       | 9987           | object   |
| 16  | WSPM     | 9995           | float64  |
| 17  | station  | 10000          | object   |  

Dari output fungsi ```.info()``` menunjukan jumlah baris dari dataset yang diambil berjumlah 10000 baris. Hal ini dilakukan karena untuk meload ±35.000 data cukup banyak, dan 10000 data sudah dirasa cukup. Selain itu, dataset di atas memiliki total kolom sejumlah 18 kolom, dengan kolom integer berjumlah 5 kolom, kolom float berjumlah 11 kolom, dan kolom object berjumlah 2 kolom

#### EDA - Deskripsi Variabel
Berikut adalah penjelasan untuk tiap tiap variabel / fitur pada dataset yang dipakai :
- `No` : Nomor baris data.  
- `year`, `month`, `day`, `hour`: Fitur waktu yang menunjukkan kapan data direkam.  
- `PM2.5`: Konsentrasi partikulat PM2.5 (µg/m³).  (**target yang akan diprediksi.**)  
- `PM10`: Konsentrasi partikulat PM10 (µg/m³).  
- `SO2`: Konsentrasi sulfur dioksida (µg/m³).  
- `NO2`: Konsentrasi nitrogen dioksida (µg/m³).  
- `CO`: Konsentrasi karbon monoksida (µg/m³).  
- `O3`: Konsentrasi ozon (µg/m³).  
- `TEMP`: Temperatur udara dalam satuan Celsius (°C).  
- `PRES`: Tekanan udara dalam satuan hektopaskal (hPa).  
- `DEWP`: Titik embun (Dew Point) dalam satuan Celsius (°C).  
- `RAIN`: Curah hujan dalam satuan milimeter (mm).  
- `wd`: Arah angin (contoh: N, NE, E), merupakan fitur kategorikal.  
- `WSPM`: Kecepatan angin dalam satuan meter per detik (m/s).  
- `station`: Nama stasiun pemantauan (Aotizhongxin).

### EDA - Missing Values, Duplicated, Outliers
##### Missing Values  
| Column | Missing Values |
|--------|----------------|
| year   | 0              |
| month  | 0              |
| day    | 0              |
| hour   | 0              |
| PM2.5  | 275            |
| PM10   | 213            |
| SO2    | 267            |
| NO2    | 308            |
| CO     | 521            |
| O3     | 473            |
| TEMP   | 8              |
| PRES   | 8              |
| DEWP   | 8              |
| wd     | 13             |
| WSPM   | 5              |
               
|Total   |       2099     |
|--------|----------------|

Dari output `.isnull().sum()` diketahui terdapat 6 kolom dengan jumlah missing values yang lumayan banyak. Karena jika ditotal berjumlah 2099, maka missing values ini tidak bisa langsung didrop begitu saja  

##### Duplicated  
|Duplicated   |       0     |
|-------------|-------------|  

Dari output `.duplicated().sum()` diketahui jumlah data yang terduplikasi pada dataset ini tidak ada, karena output menunjukan nilai 0. Oleh karena itu tidak diperlukan tindakan apapun  

##### Outliers  
![04f454e0-7482-4ea7-bc25-9a51a6abca04](https://github.com/user-attachments/assets/b25a1c4d-4142-4c34-bbae-d80d3756727a)  

Dari output program untuk mengecek Outliers di atas, terlihat beberapa kolom memiliki outlier yang nilainya memiliki jarak cukup jauh dari distribusi mayoritas data yang lainnya. Diantaranya pada fitur PM2.5, PM10, SO2, NO2, RIN, O3, dan CO. Sehingga akan dilakukan penanganan Outliers dengan menghapus data yang termasuk ke dalam ranah Outliers menggunakan IQR

#### Univariate Analysis
##### Kategorikal Fitur  
Hanya terdapat 1 fitur kategorikal pada dataset ini, yaitu fitur / kolom `wd`. Dengan distribusi sebagai berikut :
![image-1](https://github.com/user-attachments/assets/516a32c2-3d2e-42fa-8a2c-f88ff10ed0db)  

Berdasarkan gambar di atas, maka fitur `wd` memiliki distribusi data right-skewed yang berarti sebagian besar data terkonsentrasi di sisi kiri
##### Numerik Fitur
Untuk numerik fitur pada dataset ini yang dipakai berjumlah 14 fitur, yaitu fitur `year`, `month`, `day`, `hour`, `PM2.5`, `PM10`, `SO2`, `NO2`, `CO`, `O3`, `TEMP`, `PRES`, `DEWP`, `WSPM`. Distribusi tiap fitur numerik disajikan dalam diagram berikut :
![image-3](https://github.com/user-attachments/assets/20ad323f-fa33-4e7e-849b-4619a1f3d18e)  

Dari gambar di atas, terlihat bahwa mayoritas fitur numerik memiliki distribusi data right-skewed, sama dengan fitur `wd`. Lalu beberapa data lainnya berdistribusi acak maupun normal, dan hanya sedikit diantarnya yang berdistribusi left-skewed.

#### Multivariate Analysis
##### Fitur `wd` terhadap `PM2.5`
![image-4](https://github.com/user-attachments/assets/acaafd94-bd01-474b-ab56-66168045120b)  

Dari hasil gambar relasi antara fitur `wd` terhadap fitur target yaitu `PM2.5`, menunjukan bahwa data arah mata angin didominasi oleh SSE atau South-Southeast, ESE atau East-Southeast, dan SE atau Southeast. Lalu persebaran datanya tidak terlalu merata karena terdapat selisih yang cukup jauh  
##### Fitur numerik terhadap `PM2.5`
![image-5](https://github.com/user-attachments/assets/2b348a84-88f9-454b-bab6-12c16bd2954e)  
![image-6](https://github.com/user-attachments/assets/db62d967-b404-48cb-92af-94751c2fd974)  

Dilihat dari hasil pairplot dan juga heatmap, fitur `PM10` memiliki korelasi yang cukup tinggi terhadap fitur `PM2.5` dengan nilai korelasi 0.84. Lalu fitur `O3` memiliki korelasi paling rendah dengan nilai korelasi -0.06. Oleh karena itu, kedua fitur tersebut di drop
****
Kemudian terdapat fitur yang berhubungan dengan waktu yaitu fitur `year`, `month`, `day`, dan `hour`, dimana fitur ini cukup pentin untuk mendapatkan pola dari data, sehingga walaupun fitur ini memiliki korelasi yang cukup rendah, namun tidak didrop.  

Selanjutnya terdapat fitur `TEMP`, `PRES`, dan `DEWP`. Ketiga fitur ini memiliki korelasi yang cukup tinggi satu sama lain, sehingga nantinya akan dilakukan feature_engineering untuk mereduksi data.

## Data Preparation
#### Drop Fitur  
|     | Column | Non-Null Count | Dtype   |
|-----|--------|----------------|---------|
| 0   | year   | 10000          | int64   |
| 1   | month  | 10000          | int64   |
| 2   | day    | 10000          | int64   |
| 3   | hour   | 10000          | int64   |
| 4   | PM2.5  | 9725           | float64 |
| 5   | PM10   | 9787           | float64 |
| 6   | SO2    | 9733           | float64 |
| 7   | NO2    | 9692           | float64 |
| 8   | CO     | 9479           | float64 |
| 9   | O3     | 9527           | float64 |
| 10  | TEMP   | 9992           | float64 |
| 11  | PRES   | 9992           | float64 |
| 12  | DEWP   | 9992           | float64 |
| 13  | wd     | 9987           | object  |
| 14  | WSPM   | 9995           | float64 |  

Dilakukan drop fitur pada dataset, yaitu fitur `No` dan `station` karena kedua fitur tersebut tidak berpengaruh pada data, dan juga drop fitur `RAIN`. Fitur `RAIN` di drop karena terdapat 8500++ baris data pada fitur `RAIN` yang bernilai 0.0 walaupun tidak NULL.  
Output dari `main_df[main_df['RAIN'] == 0.0].value_counts().sum()` adalah `np.int64(8705)`, sehingga `RAIN` perlu di drop  

#### Handling Missing Values  
Karena hasil pengecekan missing values menunjukan terdapat cukup banyak fitur yang memiliki missing values, maka dalam dataset ini missing values tidak bisa didrop begitu saja. Maka dari itu, dilakukan imputasi menggunakan nilai **rata - rata / mean** dari fitur tersebut.  
`main_df['CO'] = main_df['CO'].fillna(main_df['CO'].mean())  
main_df['O3'] = main_df['O3'].fillna(main_df['O3'].mean())  
main_df['NO2'] = main_df['NO2'].fillna(main_df['NO2'].mean())  
main_df['SO2'] = main_df['SO2'].fillna(main_df['SO2'].mean())  
main_df['PM10'] = main_df['PM10'].fillna(main_df['PM10'].mean())  
main_df['PM2.5'] = main_df['PM2.5'].fillna(main_df['PM2.5'].mean())  
main_df.isnull().sum().sum()`  
Outpunya adalah `np.int64(42)`. Nah setelah mayoritas missing values sudah ditangani, sisa missing values yang ada dapat didrop saja karena tidak terlalu berpengaruh terhadap data  

#### Handling Outliers
Outliers pada dataset ini ditangani dengan teknik IQR yaitu pengukuran terhadap penyebaran nilai data dengan mengabaikan nilai esktrem. Berikut plot hasil dari dataset yang sudah tidak memiliki outliers  
![f4062140-1c48-4943-8b1b-4a9e4c5868a7](https://github.com/user-attachments/assets/ce007318-f83e-4812-9741-38521eb75f25)  

Dari gambar di atas, terlihat fitur fitur pada dataset sudah tidak memiliki outliers, dan ukurannya / jumlah barisnya pun sudah berkurang dengan mengecek output fungsi `.shape()`  
`(7536, 15)`

#### Fitur Engineering
##### Fitur `wd` dan `WSPM`
Fitur `wd` yang merupakan indikasi arah mata angin dan juga fitur `WSPM` yang merupakan indikasi nilai kecepatan angin memiliki keterkaitan yang cukup dekat pada kehidupan. Fitur engineering dilakukan pada kedua fitur ini karena fitur `wd` yang merupakan fitur kategorikal memiliki total 16 nilai unik. Hal ini akan cukup menyulitkan saat dilakukan Fitur Encoding karena akan meningkatkan dimensi fitur yang ada. Oleh karena itu, fitur `wd` akan dikombinasikan dengan fitur `WSPM` untuk mendapatkan 2 fitur baru yaitu vektor x dan vektor y yang mengindikasikan pergerakan arah mata angin.
| year | month | day | hour | PM2.5 | SO2  | NO2  | CO     | TEMP | PRES  | DEWP  | wind_x    | wind_y    |
|------|-------|-----|------|-------|------|------|--------|------|-------|-------|-----------|-----------|
| 2013 | 9     | 6   | 15   | 67.0  | 11.0 | 57.0 | 1100.0 | 25.6 | 1008.9| 16.4  | -1.301124 | -3.141190 |
| 2016 | 11    | 13  | 10   | 69.0  | 5.0  | 68.0 | 2200.0 | 7.9  | 1014.9| 3.7   | 0.267878  | 0.646716  |
| 2016 | 2     | 16  | 23   | 11.0  | 31.0 | 32.0 | 500.0  | 2.9  | 1022.0| -13.0 | 2.400000  | 0.000000  |
| 2013 | 12    | 27  | 2    | 15.0  | 19.0 | 40.0 | 600.0  | -3.7 | 1027.8| -18.8 | 1.400000  | 0.000000  |
| 2014 | 3     | 30  | 18   | 29.0  | 6.0  | 47.0 | 400.0  | 22.4 | 1008.9| -1.7  | -1.385819 | -0.574025 |  

##### Fitur `TEMP`, `PRES`, dan `DEWP` dengan PCA
Sepertinya yang telah disebutkan sebelumnya, ketiga fitur ini memiliki korelasi yang cukup tinggi. Sehingga dapat dilakukan _dimention reduction_. Hasil penerapan PCA pada ketiga kolom menunjukan bahwa PC1 dan PC2 sudah cukup untuk mewakiliki data sebelumnya, sehingga hanya akan digunakan PCA dengan hasil 2 komponen
| year | month | day | hour | PM2.5 | SO2  | NO2  | CO     | wind_x        | wind_y        | components_1 | components_2 |
|------|-------|-----|------|-------|------|------|--------|---------------|---------------|--------------|--------------|
| 2013 | 6     | 10  | 7    | 15.0  | 6.0  | 48.0 | 700.0  | -1.285879e-16 | -7.000000e-01 | 11.363233    | 6.415562     |
| 2014 | 6     | 6   | 5    | 96.0  | 8.0  | 67.0 | 1000.0 | 6.888302e-01  | 1.662983e+00  | 21.052108    | -0.436254    |
| 2014 | 8     | 10  | 20   | 57.0  | 3.0  | 37.0 | 500.0  | -1.836970e-16 | -1.000000e+00 | 26.284126    | -0.508193    |
| 2014 | 8     | 12  | 20   | 31.0  | 5.0  | 44.0 | 400.0  | -0.000000e+00 | 0.000000e+00  | 21.024681    | -0.392706    |
| 2015 | 4     | 29  | 9    | 128.0 | 11.0 | 88.0 | 1600.0 | -9.000000e-01 | 1.102182e-16  | 8.757015     | 3.714110     |

#### Data Splitting
Data yang ada dibagi menjadi 2 bagian yaitu **data train** dan juga **data test**. Pembagian dataset ini menggunakan perbandingan 80:20, dimana 80% dari dataset keseluruhan untuk **data train** dan 20%-nya untuk **data test**. Hasil pembagian tersebut sebagai berikut :  
- Total sampel di seluruh dataset: 7536  
- Total sampel di train dataset: 6028  
- Total sampel di test dataset: 1508  

#### Standarisasi
Teknik yang digunakan untuk digunakan Standarisasi adalah StandardScaler(). Teknik ini digunakan karena mayoritas fitur dalam dataset tidak berdistribusi normal, sehingga digunakan StandardScaler agar distribusi data berdasarkan standar deviasi menjadi bernilai 1 dan rata rata nya 0. Hasil Standarisasi ini menjadikan seluruh fitur diperlakukan setara oleh model. Berikut adalah hasil fungsi describe() setelah dilakukan Standarisasi : 
| stat | year     | month    | day      | hour     | SO2      | NO2      | CO        | wind_x   | wind_y   | components_1 | components_2 |
|------|----------|----------|----------|----------|----------|----------|-----------|----------|----------|--------------|--------------|
| count| 6028.0000| 6028.0000| 6028.0000| 6028.0000| 6028.0000| 6028.0000| 6028.0000 | 6028.0000| 6028.0000| 6028.0000    | 6028.0000    |
| mean | 2014.6506| 6.6923   | 15.9404  | 11.1413  | 11.1917  | 54.1039  | 930.0275  | 0.2176   | 0.0603   | -0.0897      | 0.0579       |
| std  | 1.1773   | 3.2540   | 8.7637   | 7.0292   | 10.5317  | 28.8335  | 585.6795  | 1.3180   | 1.2221   | 19.2643      | 5.5852       |
| min  | 2013.0000| 1.0000   | 1.0000   | 0.0000   | 0.2856   | 2.0000   | 100.0000  | -3.7879  | -3.9000  | -55.3296     | -25.7502     |
| 25%  | 2014.0000| 4.0000   | 8.0000   | 5.0000   | 3.0000   | 31.0000  | 500.0000  | -0.6467  | -0.7391  | -16.8906     | -3.2596      |
| 50%  | 2015.0000| 7.0000   | 16.0000  | 11.0000  | 7.0000   | 53.0000  | 800.0000  | 0.0707   | 0.0707   | 3.0278       | 0.5573       |
| 75%  | 2016.0000| 9.0000   | 23.0000  | 17.0000  | 17.0000  | 72.0000  | 1258.0659 | 0.9899   | 0.9192   | 17.2103      | 3.8892       |
| max  | 2017.0000| 12.0000  | 31.0000  | 23.0000  | 48.0000  | 155.0000 | 3000.0000 | 4.1000   | 3.7879   | 34.4312      | 16.1063      |

## Modeling  

1.  **Feature Selection**: Menghapus fitur `PM10` karena sangat berkorelasi dengan target `PM2.5` (redundan) dan `O3` karena korelasinya sangat rendah.
2.  **Pemisahan Data**: Membagi dataset menjadi data latih (80%) dan data uji (20%) menggunakan `train_test_split`.
3.  **Scaling Fitur**: Melakukan standardisasi pada seluruh fitur numerik menggunakan `StandardScaler` agar setiap fitur memiliki skala yang sebanding, yang penting untuk model seperti SVR dan KNN.

## Modeling  
Pada tahap ini, beberapa model machine learning dikembangkan dan dievaluasi untuk menemukan solusi terbaik dalam memprediksi tingkat **PM2.5**, sesuai dengan solution statement yang telah dirumuskan. Semua model dilatih menggunakan data latih yang telah melalui tahapan data preparation dan scaling untuk memastikan perbandingan yang adil dan akurat.
#### Algoritma
Tiga algoritma regresi yang digunakan adalah:
- Random Forest
  `RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)`
  Random Forest adalah model ensemble yang bekerja dengan membangun banyak decision tree (pohon keputusan) secara acak dan menggabungkan hasil prediksi dari semua pohon tersebut. Dengan merata-ratakan prediksi, model ini menjadi lebih stabil dan akurat. Model Random Forest yang dibuat menggunakan beberapa parameter yaitu :  
  1. **n_estimators = 50**  : ini berarti jumlah pohon keputuan / decision tree yang akan dibuat berjumlah 50 pohon. Semakin banyak, akan semakin stabil hasilnya, tapi lebih lambat.
  2. **max_depts = 16** : ini berarti maksimum kedalaman setiap pohon yang dibuat adalah 16, batasan ini mencegah pohon menjadi terlalu dalam (overfitting)
  3. **random_state = 42** : merupakan seed untuk pengacakan agar konsisten
  4. **n_jobs = -1** : bermakna agar semua core CPU yang tersedia akan digunakan  
  
  Random Forest memiliki kelebihan yaitu sangat kuat dalam menangkap pola non-linear yang kompleks, tahan terhadap overfitting, dan sering memberikan performa tinggi tanpa perlu banyak penyetelan awal. 
  Lalu kekurangan dari Random Forest adalah cenderung menjadi "kotak hitam" (black box), di mana proses pengambilan keputusannya sulit untuk diinterpretasikan secara langsung.
- Support Vector Machine / SVR
  SVR bekerja dengan menemukan sebuah hyperplane yang paling cocok untuk data, namun dengan toleransi kesalahan tertentu (margin). Tujuannya adalah agar sebanyak mungkin titik data berada di dalam margin tersebut. Model Support Vector Machine atau SVM memiliki beberapa parameter, pada projek ini digunakan 1 paramter yaitu :  
  **kernel = 'rbf'**  
  Yang berarti model akan menggunakan kernel Radial Basis Function, yang memungkinkan model untuk menangani data non-linear  
  Kelebihan SVR yaitu ia cukup efektif jika digunakan di ruang fitur berdimensi tinggi dan fleksibel dalam menangani data non-linear.
  Kekurangan SVR adalah sangat sensitif terhadap skala fitur (wajib di-scaling) dan performanya sangat bergantung pada pemilihan hyperparameter yang tepat. Proses pelatihannya bisa menjadi lambat pada dataset yang sangat besar.
- K-Neighbors Regressor / KNN
  KNN adalah algoritma sederhana yang memprediksi nilai suatu titik data baru dengan mengambil rata-rata dari nilai target `k` tetangga terdekatnya di dalam data latih. Pada model yang dibuat nilai `k` adalah :
  **n_neighbors = 10**  
  Ini berarti model akan melakukan prediksi menggunakan 10 tetangga terdekat dari data, dengan ini model tidak terlalu mudah overfitt.
  Kelebihan KNN adalah mudah dipahami dan diimplementasikan karena tidak menggunakan algoritma yang rumit.
  Kekurangannya KNN sensitif terhadap skala fitur, performanya dapat menurun pada data dengan banyak fitur (curse of dimensionality), dan proses prediksinya bisa lambat karena perlu menghitung jarak ke semua titik data latih.

#### Pemilihan Model Terbaik
Setelah ketiga model dilatih, performa masing-masing dievaluasi menggunakan data uji (test set) untuk mengukur kemampuan generalisasinya. Pemilihan model terbaik didasarkan pada metrik evaluasi yang telah ditentukan, yaitu **Mean Absolute Error (MAE)** dan **Mean Squared Error (MSE)**.
Berdasarkan hasil evaluasi (yang akan dirinci pada bagian selanjutnya), Random Forest Regressor diperkirakan sebagai model terbaik. Alasannya karena model Random Forest merupakan model yang menggunakan teknik **Ensemble** sehingga melakukan train model dengan banyak algoritma lainnya, yaitu Decision Tree.


## Evaluasi
Tahap evaluasi bertujuan untuk mengukur performa dari setiap model yang telah dilatih secara objektif. Metrik yang digunakan harus sesuai dengan tipe masalah regresi, di mana kita ingin mengukur seberapa dekat nilai prediksi model dengan nilai aktualnya. Untuk proyek ini, metrik evaluasi yang digunakan adalah **Mean Absolute Error (MAE)** dan **Mean Squared Error (MSE)**.

#### Metrik Evaluasi
##### Mean Absolute Error / MAE
MAE mengukur rata-rata dari selisih absolut antara nilai prediksi dan nilai sebenarnya. Keunggulan utama MAE adalah mudah diinterpretasikan karena berada dalam satuan yang sama dengan variabel target. Metrik ini memberikan gambaran seberapa besar rata-rata kesalahan prediksi model.  

$$MAE = \frac{1}{n} \sum_{i=1}^{n} |yi - yp|^2$$  

Dengan :
- n = jumlah dataset
- yi = nilai aktual
- yp = nilai prediksi

##### Mean Squared Error / MSE
MSE mengukur rata-rata dari kuadrat selisih antara nilai prediksi dan nilai sebenarnya. Dengan mengkuadratkan selisih, metrik ini memberikan "hukuman" yang jauh lebih besar pada kesalahan yang besar (pencilan). Semakin kecil nilai MSE, semakin baik performa model dalam meminimalkan kesalahan besar.

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (yi - yp)^2$$  

#### Hasil Evaluasi
##### MSE
![image-8](https://github.com/user-attachments/assets/016e5eb6-085d-4f57-b83d-85fe7840fc17)  

##### MAE
![image-9](https://github.com/user-attachments/assets/ca816e49-737b-4ea4-a533-74d0f4f5ce0f)  

Dari hasil evaluasi kedua metrik di atas, terlihat bahwa ternyata model yang memiliki nilai MSE dan MAE paling rendah adalah model **Support Vector Regressor (SVM)**, sehingga model SVM dapat dipilih sebagai model terbaik.
Selain itu, nilai error pada data latih dan data uji untuk setiap model tidak berbeda jauh, yang mengindikasikan bahwa model-model tersebut tidak mengalami overfitting secara signifikan. Namun, keunggulan performa SVM pada data yang belum pernah dilihat sebelumnya menjadikannya solusi paling andal untuk permasalahan ini  

#### Dampak Terhadap Business Understanding  
##### Problem Statement  
- Dengan dikembangkannya model SVR (SVM) yang memiliki nilai Test **MAE 0.038**, maka berhasil mengembangkan alat prediksi yang cukup akurat. Tingkat kesalahan yang rendah ini berarti model mampu memberikan perkiraan tingkat PM2.5 yang mendekati kenyataan, memenuhi kebutuhan akan sistem peringatan dini yang andal.  
- Meskipun model SVR perlu sedikit pemahaman lebih untuk diinterpretasikan, proses data preparation sebelumnya (seperti PCA dan analisis korelasi) secara tidak langsung mengidentifikasi faktor-faktor penting. Terbukti bahwa komponen gabungan cuaca (`TEMP`, `PRES`, `DEWP`) dan polutan lain seperti `CO` dan `NO2` memiliki pengaruh signifikan terhadap polusi di daerah tertentu  

##### Goals  
- Tujuan pertama berhasil dicapai dengan terpilihnya model SVR yang menunjukkan nilai error terendah pada data test. Hasil inferensi singkat (**y_true=42.0, prediksi_SVM=49.2**) menunjukkan bahwa meskipun tidak sempurna, prediksi SVR adalah yang paling mendekati nilai sebenarnya dibandingkan model lain  
- Tujuan kedua tercapai, dimana analisis pada tahap data understanding dan data preparation berhasil mengonfirmasi bahwa selain polutan lain, faktor meteorologi adalah pendorong utama tingkat **PM2.5**.  

##### Solution Statement  
- Solusi pertama terbukti berdampak, dengan membandingkan tiga model berbeda, dapat membuktikan secara kuantitatif bahwa SVR adalah pilihan terbaik, mengalahkan Random Forest dan KNN dengan selisih performa yang cukup jelas terlihat. Tanpa perbandingan ini, mungkin akan dipilih model yang kurang optimal.  
- Solusi kedua bisa dibilang cukup krusial, karena penggunaan metrik MAE dan MSE memberikan dasar diambilnya setiap keputusan. Metrik ini secara tegas mengonfirmasi bahwa solusi yang diajukan (model SVR) tidak hanya berfungsi, tetapi juga merupakan solusi terbaik di antara kandidat yang ada untuk mencapai goals yang telah ditetapkan
