# Bu kod Google Colab'a uygun oqalcak şekilde hazırlanmıştır.

# FS-ANN ve Klasik MLP: Gogus Kanseri Karsilastirmasi

Bu depo, tez calismasinda anlatilan **FS-ANN** (T-Norm / T-Conorm ve Gauss uyelik fonksiyonlu, aciklanabilir yapay sinir agi) ile klasik bir cok katmanli algilayicinin (MLP) Wisconsin Gogus Kanseri veri seti uzerinde karsilastirmasini icerir.

## Dosyalar
- `fsann_gogus_kanseri.py` : Veri on isleme, iki modelin egitimi, test sonuclari ve FS-ANN agirliklarinin yorumlanmasi.

## Gereksinimler
Python 3.9+ ve su paketler:

```
pip install torch scikit-learn numpy
```

## Calistirma
```
python fsann_gogus_kanseri.py
```
Veri seti `scikit-learn` ile gelir (`load_breast_cancer`), ayrica indirmeye gerek yoktur.
Rastgelelik `torch.manual_seed(42)` ve `np.random.seed(42)` ile sabitlenmistir.

## Cikti
- Her model icin test dogrulugu ve F1 skoru
- Klasik MLP agirlik matrisinden bir ornek
- FS-ANN icin oznitelik agirliklari (W), yuzde etkileri ve ogrenilen Gauss parametreleri (merkez c, sigma)
