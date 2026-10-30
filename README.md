# Bu kod Google Colab'a uygun oqalcak şekilde hazırlanmıştır.
# FS-ANN ve Klasik MLP: Gogus Kanseri Karsilastirmasi

Bu depo, tez calismasinda anlatilan **FS-ANN** modelini ve klasik bir cok katmanli algilayici (MLP) ile
Wisconsin Gogus Kanseri veri seti uzerindeki karsilastirmasini icerir.

FS-ANN, girdi olarak [0,1] araligindaki uyelik derecelerini (bir bulanik esnek kumenin uyelik matrisinin satiri) alir.
Noronun carpma ve toplama islemleri yerine **Lukasiewicz T-norm** (`max(0, x + w - 1)`) ve **Lukasiewicz T-conorm**
(`min(1, toplam)`) kullanilir; aktivasyon fonksiyonu **Gauss uyelik fonksiyonudur**. Cikti [0,1] araliginda kesin bir sayidir
(`>= 0.5` ise 1. sinif).

## Dosyalar
- `fsann_ileri_besleme.py` : FS-ANN'in ileri besleme hesabi (saf Python) ve kucuk bir sayisal ornek.
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
python fsann_ileri_besleme.py
python fsann_gogus_kanseri.py
python tekrarli_degerlendirme.py
```
Veri seti `scikit-learn` ile gelir (`load_breast_cancer`), ayrica indirmeye gerek yoktur.
