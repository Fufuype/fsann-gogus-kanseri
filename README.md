# Bu kod Google Colab'a uygun oqalcak şekilde hazırlanmıştır.

# FS-ANN ve Klasik MLP: Gogus Kanseri Karsilastirmasi

Bu depo, tez calismasinda anlatilan **FS-ANN** (T-Norm / T-Conorm ve Gauss uyelik fonksiyonlu, aciklanabilir yapay sinir agi)
ile klasik bir cok katmanli algilayicinin (MLP) Wisconsin Gogus Kanseri veri seti uzerinde karsilastirmasini icerir.

## Dosyalar
- `fsann_gogus_kanseri.py` : Tek bolunmede (seed=42) iki modelin egitimi, test sonuclari ve FS-ANN agirliklarinin yorumlanmasi.
- `tekrarli_degerlendirme.py` : Ayni islem 10 farkli egitim/test bolunmesinde tekrarlanir; ortalama, standart sapma ve Wilcoxon testi yazdirilir.

## Veri sizintisi
Ozellik secimi, olcekleme ve yon hizalamasi (korelasyon isareti) yalnizca egitim verisinden ogrenilir; test verisine sadece uygulanir.

## Gereksinimler
Python 3.9+ ve su paketler:

```
pip install torch scikit-learn numpy scipy
```

## Calistirma
```
python fsann_gogus_kanseri.py
python tekrarli_degerlendirme.py
```
Veri seti `scikit-learn` ile gelir (`load_breast_cancer`), ayrica indirmeye gerek yoktur.
