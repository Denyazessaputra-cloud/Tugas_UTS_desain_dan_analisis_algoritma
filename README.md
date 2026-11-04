Nama: Muhammad Deny Azes Saputra
Kelas: TI B
NIM:TI22250028

Penjelasan Kode Tugas UTS Modeling: Prediksi Kelulusan Mahasiswa
Di tugas ini saya membuat model machine learning untuk memprediksi apakah seorang mahasiswa akan lulus atau gagal. Langkah kerjanya mengikuti alur CRISP-DM: memahami data, membersihkan dan mengeksplorasinya, menyiapkan data, membuat model, mengevaluasi, lalu mencoba model pada data baru.

Tahap 1: Memahami Data
Pertama, saya mengimpor library yang dibutuhkan: pandas dan numpy untuk mengolah data, matplotlib dan seaborn untuk grafik, serta train_test_split untuk membagi data.Setelah itu dataset original.csv dibaca dan nama kolomnya dirapikan dari spasi yang tidak perlu. Saya juga membuang baris dan kolom yang isinya kosong semua. Hasilnya, ukuran data berubah dari 87 baris dan 22 kolom menjadi 80 baris dan 20 kolom. Jadi ada 7 baris dan 2 kolom yang memang tidak berisi apa-apa.
Dari head() dan info() terlihat bahwa ada 80 mahasiswa dengan 20 kolom, semuanya bertipe angka (float64). Hanya kolom # of Absence yang punya data kosong, yaitu 2 nilai.
Lalu saya hitung jumlah mahasiswa per kelas: 68 lulus dan 12 gagal. Artinya data ini tidak seimbang, sekitar 85% lulus dan hanya 15% gagal. Ini jadi catatan penting karena model cenderung mudah menebak “lulus” terus, sehingga saya harus berhati-hati dalam memilih metrik dan pengaturan model.

Tahap 2: Eksplorasi dan Pembersihan Data
Saya mengecek data kosong, data duplikat, serta baris dan kolom yang kosong total. Hasilnya nol semua. Ada satu hal yang perlu dicatat: sel ini menunjukkan tidak ada data kosong padahal info() tadi menemukan 2 nilai kosong. Ini kemungkinan karena sel dijalankan tidak berurutan, terlihat dari nomor sel seperti In [35] dan In [12].
Setelah itu saya membuat beberapa grafik untuk melihat pola data:
Histogram umur: kebanyakan mahasiswa berumur 19 sampai 25 tahun. Ada sedikit yang jauh lebih tua, sekitar 30 sampai 45 tahun.
Boxplot total jam belajar: ada beberapa outlier, yang paling ekstrem sekitar 175 jam.
Boxplot kehadiran berdasarkan kelulusan: mahasiswa yang lulus kehadirannya lebih tinggi, median sekitar 83%, sedangkan yang gagal sekitar 54%. Jadi kehadiran jelas berhubungan dengan kelulusan.
Heatmap korelasi: fitur kehadiran dan aktivitas belajar di modul online cukup berkaitan dengan hasil akhir. Beberapa fitur juga saling mirip, misalnya TUNDRA MSOA dan TUNDRA LSOA (korelasi 1.00) serta # of presence dan # of Absence (korelasi -0.95). Fitur yang terlalu mirip seperti ini bisa mengganggu model, dan itu yang nanti saya tangani.
Saya juga memeriksa kolom Quintile. Nilai Quintile seharusnya 1 sampai 5, tetapi ada beberapa nilai 0 (2 sampai 4 data per kolom). Nilai 0 ini janggal, jadi saya anggap sebagai data yang tidak valid.
Untuk outlier, saya pakai metode IQR. Outlier ditemukan pada umur (6 data), jarak ke kampus (7), jumlah login (2), total jam belajar (3), dan beberapa kolom lain. Outlier ini tidak saya hapus karena datanya memang sudah sedikit, dan sayang kalau dibuang.
Terakhir, 2 data kosong di # of Absence saya isi dengan median (2.0). Saya pilih median karena tidak terpengaruh oleh nilai ekstrem.

Tahap 3: Menyiapkan Data
Saya membuat salinan data (data_model) supaya data asli tetap aman. Lalu nilai 0 di kolom Quintile saya ubah menjadi kosong (NaN), supaya nanti diisi otomatis oleh imputer di dalam pipeline.
Kemudian saya membuang dua fitur yang informasinya sudah terwakili fitur lain:
TUNDRA LSOA Quintile, karena hampir sama persis dengan versi MSOA.
Percent Attended, karena itu hasil hitungan dari jumlah hadir dan tidak hadir.
Setelah itu tersisa 17 fitur. Fitur (X) dan target (y) saya pisahkan, lalu data dibagi 80:20 dengan stratify=y agar perbandingan lulus dan gagal tetap sama di kedua bagian. Hasilnya 64 data latih (54 lulus, 10 gagal) dan 16 data uji (14 lulus, 2 gagal).

Tahap 4: Membuat Model
Saya mencoba dua model, keduanya dibuat dalam bentuk Pipeline supaya langkah pra-pemrosesan ikut terhitung saat validasi dan tidak ada kebocoran data.
Logistic Regression: mengisi data kosong dengan median, menyeragamkan skala dengan RobustScaler, lalu klasifikasi dengan class_weight='balanced' agar kelas “gagal” yang jumlahnya sedikit tetap diperhatikan.
Random Forest: mengisi data kosong dengan median lalu klasifikasi, juga dengan class_weight='balanced'. Parameternya saya cari dengan GridSearchCV dan hasil terbaiknya max_depth=5, min_samples_split=4, dan n_estimators=100.
Kedua model diuji dengan Stratified 5-Fold Cross Validation memakai metrik F1-score. Saya pilih F1 karena data tidak seimbang, jadi akurasi saja bisa menyesatkan.
Model	Rata-rata F1 (Cross Validation)
Logistic Regression	0.5419
Random Forest	0.4933
Menurut cross validation, Logistic Regression sedikit lebih unggul.

Tahap 5: Evaluasi pada Data Uji
Setelah itu kedua model diuji pada 16 data uji:
Model	Accuracy	Precision	Recall	F1
Logistic Regression	0.8125	0.3333	0.5	0.40
Random Forest	0.9375	1.0000	0.5	0.6667
Dari confusion matrix:
Logistic Regression: 12 mahasiswa lulus ditebak benar, 2 lulus salah ditebak gagal, 1 gagal salah ditebak lulus, dan 1 gagal ditebak benar.
Random Forest: 14 mahasiswa lulus ditebak benar semua, 1 gagal terlewat (ditebak lulus), dan 1 gagal ditebak benar.
Saya juga melihat feature importance dari Random Forest untuk tahu fitur mana yang paling berpengaruh. Tiga teratas adalah % of Average Hours in Module Area (0.26), Count of Module Area Logins (0.17), dan Total Hours in Module Area (0.12), disusul # of Absence. Jadi yang paling menentukan adalah seberapa aktif mahasiswa belajar di modul online dan seberapa rajin hadir. Faktor latar belakang seperti umur dan quintile wilayah pengaruhnya jauh lebih kecil.

Tahap 6: Uji Prediksi pada Data Baru
Model terbaik dipilih berdasarkan F1 pada data uji, jadi yang terpilih adalah Random Forest (0.6667 dibanding 0.40).
Untuk mencobanya, saya membuat 5 data mahasiswa fiktif secara acak dengan default_rng(42), dengan rentang nilai yang disesuaikan dengan data asli. Jumlah ketidakhadiran dihitung dari 12 - jumlah kehadiran. Model lalu memprediksi apakah tiap mahasiswa lulus atau gagal, lengkap dengan persentase probabilitas gagalnya.

Kesimpulan
Setelah dibersihkan, dataset berisi 80 mahasiswa dengan 17 fitur. Datanya tidak seimbang: 68 lulus dan 12 gagal.
Faktor yang paling menentukan kelulusan adalah keaktifan di modul pembelajaran online (persentase jam belajar, jumlah login, total jam) dan kehadiran. Faktor demografis pengaruhnya kecil.
Pada data uji, Random Forest tampil lebih baik (accuracy 93,75% dan F1 0,67) dibanding Logistic Regression (accuracy 81,25% dan F1 0,40).
Hasil ini menunjukkan bahwa machine learning bisa dimanfaatkan sebagai sistem peringatan dini untuk menemukan mahasiswa yang berisiko gagal, supaya bisa dibantu lebih awal.

Keterbatasan
Hasil cross validation dan data uji tidak sejalan. Di cross validation Logistic Regression lebih unggul, tetapi di data uji justru Random Forest. Penyebabnya, data uji hanya 16 data dengan 2 mahasiswa gagal, jadi satu tebakan saja bisa mengubah nilai metrik jauh. Karena itu, kesimpulan “Random Forest paling baik” belum terlalu kuat.
Recall kelas gagal baru 0.5. Kedua model baru bisa menemukan separuh mahasiswa yang benar-benar gagal. Untuk peringatan dini, ini perlu ditingkatkan karena justru mahasiswa berisiko inilah yang paling ingin kita tangkap.
Data di Tahap 6 hanya data acak, bukan data mahasiswa sungguhan. Jadi tahap itu hanya menunjukkan cara kerja model, bukan membuktikan seberapa akurat model tersebut. Ke depan, hasil bisa lebih meyakinkan kalau datanya ditambah, atau evaluasi dilakukan dengan cross validation pada seluruh data.
