# KlasifikasiNilam

KlasifikasiNilam adalah software backend yang berfungsi mengklasifikasikan citra daun nilam. Ekstraksi fitur yang digunakan adalah kombinasi dari LBP, Convex Hull, dan Morphology, dan pengenalan data menggunakan Extreme Learning Machine.

## Instalasi (Windows)

```cmd
C:\KlasifikasiNilam\> python -m venv venv
C:\KlasifikasiNilam\> venv\Scripts\activate
(venv) C:\KlasifikasiNilam\> pip install -r requirements.txt
```

Untuk keluar dari venv:

```cmd
(venv) C:\KlasifikasiNilam\> venv\Scripts\deactivate.bat
C:\KlasifikasiNilam\>
```

## Instalasi (Linux)

```bash
user@localhost:~/KlasifikasiNilam$ python -m venv venv
user@localhost:~/KlasifikasiNilam$ source venv/bin/activate
(venv) user@localhost:~/KlasifikasiNilam$ pip install -r requirements.txt
```

Untuk keluar dari venv:
```bash
(venv) user@localhost:~/KlasifikasiNilam$ deactivate
user@localhost:~/KlasifikasiNilam$ 
```

## Kebutuhan

* Minimal menggunakan Python versi 3.x
* Memahami penggunaan virtualenvironment (venv)

Note: Gunicorn hanya bisa digunakan di OS berbasis Linux.

## Penggunaan (Windows)

```cmd
(venv) C:\KlasifikasiNilam\> python launch_backend.py
```

## Penggunaan (Linux)

```bash
(venv) user@localhost:~/KlasifikasiNilam$ gunicorn --bind 0.0.0.0:5000 wsgi:app
```

## Mengubah pengaturan

Pengaturan seperti path dan jumlah hidden node (untuk ELM) dapat ditemukan sebagai variabel pada file start.py di dalam folder algorithm. Jika ingin mengaturnya melalui notepad, dapat menggunakan kode di bawah:

```cmd
(venv) C:\KlasifikasiNilam\> cd algorithm
(venv) C:\KlasifikasiNilam\algorithm> notepad start.py
```

## Mengakses aplikasi

### Hello world.

Secara default, flask akan membuka port 5000 di sistem Anda.
Untuk mengecek, silakan buka:

```
http://localhost:5000/
```
Jika semua sudah benar, teks di bawah akan muncul di browser.
```
Hello.
```

Untuk mengetahui format json yang akan dikeluarkan oleh software ini, dapat dibuka:
```
http://localhost:5000/jsontest
```
### Spesifikasi POST request (terutama untuk Postman)

* Pastikan "content-type" berupa "multipart/form-data"
* Pastikan pada body, gunakan jenis "form-data"
* Pada body, "key" harus string bernama 'image' dengan tipe file, dan "value" berupa file yang ingin diprediksi.

---
KlasifikasiNilam adalah repository untuk kode yang digunakan saat PKL KJFD MGM 2020, FILKOM Universitas Brawijaya.
