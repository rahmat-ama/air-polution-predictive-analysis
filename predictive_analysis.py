
# # Air Polution Predictive Analysis

# ### Deskripsi
# Proyek ini dibuat untuk melakukan sebuah analisis prediktif terhadap topik polusi udara di suatu daerrah. Dengan menggunakan model regresi machine learning, berdasarkan fitur tertentu yang akan dijadikan sebagai target, lalu hasil prediksi akan menemukan nilai tertentu untuk dijadikan acuan bagaimana keadaan polusi pada bulan, minggu, hari, dan jam tertentu. Dengan hasil prediksi ini, dapat memberikan informasi kepada masyarakat tentang kapan perlu menggunakan masker dengan peringatan yang lumayan serius kepada masyarakat, ataupun himbauan untuk mengurangi kegiatan / berolahraga diluar terlebih dahulu untuk hari maupun jam tertentu. Dimana polusi udara ini masih perlu menjadi perhatian khususnya di kota-kota besar.

# ## 1. Import library

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error

# ## 2. Data Understanding

# Dataset yang digunakan merupakan dataset dari kaggle dengan link di bawah ini
# [PRSA_Data_Aotizhongxin 2013 - 2017](https://www.kaggle.com/datasets/shaviranurfadhilla/prsa-data-aotizhongxin-2013-2017)

# ### Data Loading

air_dataset = 'PRSA_Data_Aotizhongxin.csv'
np.random.seed(42)
total_rows = 35064
rows_to_read = 10000
rows_to_skip = np.random.choice( np.arange(1, total_rows + 1),
    size=total_rows - rows_to_read,replace=False)
main_df = pd.read_csv(air_dataset, skiprows=rows_to_skip)
main_df.sample(5)

# Data yang digunakan proyek ini hanya mengambil data dengan total kolom 10000 baris dari dataset asli. Untuk lingkup percobaan sudah cukup untuk membangun model machine learning

# ### EDA - Deskripsi variabel

main_df.info()

# Dari dataframe utama, didapatkan informasi bahwa dataset ini memiliki 18 kolom. Dengan deskripsi sebagai berikut :  
# - No : Penomoran tiap baris data
# - year : Tahun data tersebut
# - month : Bulan data tersebut
# - day : Hari data tersebut
# - hour : Jam data tersebut
# - PM2.5 : Particulate matter 2.5, partikel udara dengan ukuran <= 2.5 mikrometer
# - PM10 : Particulate matter 10, partikel udara dengan ukuran <= 10 mikrometer
# - SO2 : Sulfur dioksida, gas yang terbentuk ketika sulfur bereaksid dengan oksigen
# - NO2 : Nitrogen dioksida, polutan udara yang dihasilkan bahan bakar
# - CO : Karbon monoksida, terbentuk ketika bahan bakar dengan karbon dibakar tanpa cukup oksigen
# - O3 : Ozon, gas yang terdiri dari 3 atom oksigen
# - TEMP : Nilai temperature / suhu
# - PRES : Nilai tekanan udara
# - DEWP : Dew point, titik embun (nilai kelembapan udara)
# - RAIN : Intensitas hujan
# - wd : Wind direction, arah angin berhembus
# - WSPM : Wind speed, kecepatan angin
# - station : lokasi pemantauan  
# 
# Dimana fitur yang akan digunakan untuk menjadi target adalah PM2.5 yang menunjukan indikasi seberapa banyak partikel di udara yang ukurannya sangat kecil sehingga mudah mencemari udara

main_df.describe(include='all')

# Didapatkan macam macam parameter data numerik maupun kategorikal berdasarkan output di atas

main_df.drop(["No", "station"],axis=1,inplace=True)

# Karena kolom 'No' dan 'station' tidak relevan untuk digunakan pada model machine learning, maka dilakukan drop kolom tersebut

main_df[main_df['RAIN'] == 0.0].value_counts().sum()

# Di lain sisi, ada kolom RAIN yang memiliki nilai 0.0 dengan dominasi sangat tinggi. Maka dari itu bisa dibilang kolom RAIN ini tidak memiliki pengaruh yang signifikan terhadap keseluruhan data, sehingga dapat di drop saja

main_df.drop("RAIN", axis=1, inplace=True)

# ### EDA - Missing values

# ##### Missing values

missing_val = main_df.isnull().sum()
print(missing_val.sum())
print(missing_val)

# Dari pengecekan missing values, didapatkan 2099 missing values pada data. Lalu kolom dengan missing values yang cukup banyak adalah O3, CO, NO2, SO2, PM10, dan PM2.5

main_df['CO'] = main_df['CO'].fillna(main_df['CO'].mean())
main_df['O3'] = main_df['O3'].fillna(main_df['O3'].mean())
main_df['NO2'] = main_df['NO2'].fillna(main_df['NO2'].mean())
main_df['SO2'] = main_df['SO2'].fillna(main_df['SO2'].mean())
main_df['PM10'] = main_df['PM10'].fillna(main_df['PM10'].mean())
main_df['PM2.5'] = main_df['PM2.5'].fillna(main_df['PM2.5'].mean())
main_df.isnull().sum().sum()

# Setelah dilakukan imutasi dengan menggunakan mean pada kolom yang memiliki cukup banyak missing values, keseluruhan missing values menjadi 42. Ini dapat didrop saja karena tidak terlalu berpengaruh

main_df.dropna(inplace=True)
main_df.isnull().sum().sum()

# ##### Duplicated

print(main_df.duplicated().sum())

# Dari pengecekan duplikasi data, terlihat hasil menunjukan 0 yang berarti tidak ada data yang terduplikasi

# ##### Outliers

num_cols = main_df.select_dtypes(include='number').columns
fig, axes = plt.subplots(4, int(np.ceil(len(num_cols) / 4)), figsize=(4 * 5, 4 * 5))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    sns.boxplot(x=main_df[col], ax=axes[i])
    axes[i].set_title(f'Boxplot {col}')
    axes[i].set_xlabel('')
    axes[i].grid(True)
for j in range(len(num_cols), len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()

# Dari hasil pengecekan outlier di atas, terlihat beberapa kolom memiliki cukup banyak dan diantaranya ada outlier yang cukup jauh seperti kolom PM2.5, PM10, SO2, NO2, RIN, O3, dan CO

Q1 = main_df[num_cols].quantile(0.25)
Q3 = main_df[num_cols].quantile(0.75)
IQR = Q3 - Q1
filter_outliers = ~((main_df[num_cols] < (Q1 - 1.5 * IQR)) |
                    (main_df[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
main_df = main_df[filter_outliers]
main_df.shape

num_cols = main_df.select_dtypes(include='number').columns
fig, axes = plt.subplots(4, int(np.ceil(len(num_cols) / 4)), figsize=(4 * 5, 4 * 5))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    sns.boxplot(x=main_df[col], ax=axes[i])
    axes[i].set_title(f'Boxplot {col}')
    axes[i].set_xlabel('')
    axes[i].grid(True)
for j in range(len(num_cols), len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()

# Hasil dari penghapusan baris yang mengandung outlier menjadikan total keseluruhan baris pada dataset menjadi 7536 baris

# ### EDA - Univariate analysis

cat_cols = main_df.select_dtypes(include='object').columns

cat_feat = cat_cols[0]
count_cat_feat = main_df[cat_feat].value_counts()
percent_cat_feat = 100*main_df[cat_feat].value_counts(normalize=True)
cat_df = pd.DataFrame({'jumlah sampel':count_cat_feat, 'persentase':percent_cat_feat.round(1)})
print(cat_df)
count_cat_feat.plot(kind='bar', title=cat_feat);

# Univariate analysis pada kategorikal fitur yaitu kolom 'wd' memperlihatkan distribusi right skewed untuk kolom tersebut. Dimana ws / arah angin 3 terbanyak didominasi oleh NE (North East), ENE (East-Northeast), dan SW (South-West)

main_df.hist(bins=50, figsize=(20,15))
plt.show()

# Untuk distribusi data dari numerikal fitur menunjukan mayoritas fitur memiliki distribusi right skewed, lalu beberapa lainnya berdistribusi acak dan mendekati normal, lalu yang paling sedikit adalah distribusi left skewed.

# ### EDA - Multivariate analysis

cat_feat_list = cat_cols.to_list()

for col in cat_feat_list:
  sns.catplot(x=col, y="PM2.5", kind="bar", dodge=False, height = 4, aspect = 3,  data=main_df, palette="Set3")
  plt.title("Rata-rata 'PM2.5' Relatif terhadap - {}".format(col))

# Untuk multivariate analysis pada kategorikal fitur yaitu kolom 'wd' menunjukan relasi pada PM2.5 didominasi oleh nilai SSE, ESE, dan SE

sns.pairplot(main_df, diag_kind = 'kde')

plt.figure(figsize=(10, 8))
correlation_matrix = main_df[num_cols].corr().round(2)
 
# Untuk menge-print nilai di dalam kotak, gunakan parameter anot=True
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.title("Correlation Matrix untuk Fitur Numerik ", size=20)

# Dari hasil visualisasi korelasi antar fitur. Terlihat beberapa fitur memiliki berbagai macam nilai korelasi terhadap fitur PM2.5. Keputusan saat ini adalah melakukan drop untuk PM10 karena korelasi yang cukup tinggi dan mendrop O3 karena korelasi yang rendah.  
# Lalu untuk fitur waktu (year, month, day, hour) tidak di drop karena dapat berisi informasi pola data. Dan fitur TEMP, PRES, dan DEWP , lalu fitur wd dan WSPM akan dilakukan feature engineering nantinya

main_df.drop(['PM10', 'O3'], inplace=True, axis=1)
main_df.head()

# ## 3. Data Preparation

# ### Fitur Engineering

# ##### wd dan WSPM

direction_map = {
    'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
    'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
    'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
    'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
}
main_df['wind_deg'] = main_df['wd'].map(direction_map)
main_df['wind_rad'] = np.deg2rad(main_df['wind_deg'])

main_df['wind_x'] = main_df['WSPM'] * np.cos(main_df['wind_rad'])
main_df['wind_y'] = main_df['WSPM'] * np.sin(main_df['wind_rad'])
main_df = main_df.drop(['wd', 'WSPM', 'wind_deg', 'wind_rad'], axis=1)

main_df.sample(5)

# Dilakukan penggabungan antara fitur wd dan WSPM, karena jika dilakukan Encoding fitur wd, dimana fitur wd memiliki banyak nilai unik, maka akan menghasilkan banyak kolom tambahan. Di sisi lain, fitur WSPM yang merupakan kecepatan angin dapat dikombinasikan dengan fitur ws untuk membuat komponen Vekto X dan Vektor Y yang merepresentasikan pergerakan 4 arah mata angin. Sehingga tidak perlu menambah banyak kolom baru

# TEMP, PRES, DEWP dengan PCA

pca = PCA(n_components=3, random_state=123)
pca.fit(main_df[['TEMP', 'PRES', 'DEWP']])
princ_comp = pca.transform(main_df[['TEMP', 'PRES', 'DEWP']])
pca.explained_variance_ratio_.round(3)

# Hasil penerapan PCA pada ketiga kolom menunjukan bahwa PC1 dan PC2 sudah cukup untuk mewakiliki data sebelumnya, sehingga hanya akan digunakan PCA dengan hasil 2 komponen

pca = PCA(n_components=2, random_state=123)
pca.fit(main_df[['TEMP', 'PRES', 'DEWP']])
princ_comp = pca.transform(main_df[['TEMP', 'PRES', 'DEWP']])
main_df[['components_1', 'components_2']] = princ_comp
main_df.drop(['TEMP', 'PRES', 'DEWP'], axis=1, inplace=True)
main_df.sample(5)

# ### Data Splitting

X = main_df.drop(["PM2.5"],axis =1)
y = main_df["PM2.5"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print(f'Total sampel di seluruh dataset: {len(X)}')
print(f'Total sampel di train dataset: {len(X_train)}')
print(f'Total sampel di test dataset: {len(X_test)}')

# Data dibagi menjadi data train dan data test dengan pembagian 80% train dan 20% test. Lalu hasilnya adalah 6028 baris data digunakan untuk data train dan 1508 digunakan untuk data test.

# ### Standarisasi 

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train.describe().round(4)

# Hasil standarisasi menunjukan bahwa keseluruhan data sudah memiliki rata rata 0 dan standar deviasi sangat mendekati 1

# ## 4. Modeling

models = pd.DataFrame(index=['train_mse', 'test_mse', 'train_mae', 'test_mae'], 
                      columns=['Random Forest', 'SVM', 'KNN'])

# ### Random Forest

RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
RF.fit(X_train_scaled, y_train)
models.loc['train_mse', 'Random Forest'] = mean_squared_error(y_true=y_train, y_pred=RF.predict(X_train_scaled))
models.loc['train_mae', 'Random Forest'] = mean_absolute_error(y_true=y_train, y_pred=RF.predict(X_train_scaled))

# Pertama digunakan algoritma random forest, algoritma ini digunakan karena cukup mampu menangkap hubungan non-linear pada data, dan juga merupakan model ensemble yaitu model yang menggabungkan beberapa model sederhana sehingga diharapakan dapat menghasilkan nilai prediksi yang akurat

# ### SVM

SVM = SVR(kernel='rbf')
SVM.fit(X_train_scaled, y_train)
models.loc['train_mse', 'SVM'] = mean_squared_error(y_true=y_train, y_pred=SVM.predict(X_train_scaled))
models.loc['train_mae', 'SVM'] = mean_absolute_error(y_true=y_train, y_pred=SVM.predict(X_train_scaled))

# Algoritma yang kedua digunakan adalah algoritma SVM. Algoritma SVM cukup efektif dalam menangkap pola yang cenderung rumit dari data. Lalu jika dapat mengoptimalkan hyperparameter yang digunakan maka hasil yang baik bisa didapatkan

# ### KNN

KNN = KNeighborsRegressor(n_neighbors=10)
KNN.fit(X_train_scaled, y_train)
models.loc['train_mse', 'KNN'] = mean_squared_error(y_true=y_train, y_pred=KNN.predict(X_train_scaled))
models.loc['train_mae', 'KNN'] = mean_absolute_error(y_true=y_train, y_pred=KNN.predict(X_train_scaled))

# Terakhir adalah menggunakan algoritma KNN. Algoritma ini merupakann algoritma yang cukup sederhana dan mudah dipahami. Sehingga tidak rumit untuk melakukan interpretasi

# ## Model Evaluation

# Evaluasi model menggunakan Mean Squared Error atau MSE dan Mean Absolute Error atau MAE. MAE memberikan rata rata nilai absolut selisih antara nilai aktual dan prediksi, sedangkan MSE akan melakukan rata rata dari selisih yang dikuadratkan antara nilai aktual dan nilai prediksi, jadi MSE memiliki bobot yang jauh lebih besar terhadap kesalahan yang besar

mse = pd.DataFrame(columns=['mse_train', 'mse_test'], index=['RF', 'SVM', 'KNN'])
mae = pd.DataFrame(columns=['mae_train', 'mae_test'], index=['RF', 'SVM', 'KNN'])
model_dict = {'RF': RF, 'SVM': SVM, 'KNN': KNN}
for name, model in model_dict.items():
    mse.loc[name, 'mse_train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train))/1e3 
    mse.loc[name, 'mse_test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))/1e3
    mae.loc[name, 'mae_train'] = mean_absolute_error(y_true=y_train, y_pred=model.predict(X_train))/1e3 
    mae.loc[name, 'mae_test'] = mean_absolute_error(y_true=y_test, y_pred=model.predict(X_test))/1e3


mse

mae

fig, ax = plt.subplots()
mse.sort_values(by='mse_test', ascending=True).plot(kind='bar', ax=ax, zorder=3)
ax.grid(zorder=0)

fig, ax = plt.subplots()
mae.sort_values(by='mae_test', ascending=True).plot(kind='bar', ax=ax, zorder=3)
ax.grid(zorder=0)

# Hasil evaluasi model terhadap data test dilihat dari visualisasinya menunjukan bahwa model yang memiliki nilai MSE dan MAE terendah adalah model SVM. Yaitu dengan nilai MSE = 2,54 dan nilai MAE = 0,038. 

prediksi = X_test.iloc[10:11].copy()
pred_dict = {'y_true':y_test[:1]}
for name, model in model_dict.items():
    pred_dict['prediksi_'+name] = model.predict(prediksi).round(1)
pd.DataFrame(pred_dict)

# Dilihat dari inferensi singkat ketiga model di atas pun menunjukan bahwa model SVM dapat melakukan prediksi yang paling baik. Yaitu ketika nilai aktualnya adalah 42.0, nilai prediksi yang paling mendekati adalah nilai prediksi dari SVM yaitu 49.2


