# 10 farkli egitim/test bolunmesinde klasik MLP ve FS-ANN karsilastirmasi
# (fsann_gogus_kanseri.py ile ayni veri on isleme; sizinti yok)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import wilcoxon
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score

from fsann_gogus_kanseri import StandardMLP, FSANN_Fuzzy


def egit(model, X_train, y_train, X_test, y_test, epoch_sayisi, lr):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for _ in range(epoch_sayisi):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        preds = (model(X_test) >= 0.5).int().numpy()
    y_true = y_test.numpy()
    return accuracy_score(y_true, preds), f1_score(y_true, preds, zero_division=0)


def tek_calistirma(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = load_breast_cancer()
    X_raw, y = data.data, 1 - data.target

    # Once egitim/test ayrimi; sonraki tum adimlar yalnizca egitim verisinden ogrenilir
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(X_raw, y, test_size=0.2, random_state=seed)
    selector = SelectKBest(score_func=f_classif, k=5)
    X_tr_sec = selector.fit_transform(X_tr_raw, y_tr)
    X_te_sec = selector.transform(X_te_raw)
    scaler = MinMaxScaler()
    X_tr_m = scaler.fit_transform(X_tr_sec)
    X_te_m = scaler.transform(X_te_sec)

    X_tr_f, X_te_f = X_tr_m.copy(), X_te_m.copy()
    for i in range(X_tr_f.shape[1]):
        if np.corrcoef(X_tr_f[:, i], y_tr)[0, 1] < 0:
            X_tr_f[:, i] = 1.0 - X_tr_f[:, i]
            X_te_f[:, i] = 1.0 - X_te_f[:, i]

    T = lambda a: torch.tensor(a, dtype=torch.float32)
    y_tr_t, y_te_t = T(y_tr).unsqueeze(1), T(y_te).unsqueeze(1)
    mlp = egit(StandardMLP(5), T(X_tr_m), y_tr_t, T(X_te_m), y_te_t, 800, 0.01)
    fsann = egit(FSANN_Fuzzy(5), T(X_tr_f), y_tr_t, T(X_te_f), y_te_t, 800, 0.05)
    return mlp, fsann


def main():
    sonuc = np.array([[*m, *f] for m, f in (tek_calistirma(s) for s in range(10))])
    acc_m, f1_m, acc_f, f1_f = sonuc.T
    print("Model        Dogruluk (ort +- sd)     F1 (ort +- sd)")
    print(f"Klasik MLP   %{100*acc_m.mean():.2f} +- {100*acc_m.std(ddof=1):.2f}      {f1_m.mean():.4f} +- {f1_m.std(ddof=1):.4f}")
    print(f"FS-ANN       %{100*acc_f.mean():.2f} +- {100*acc_f.std(ddof=1):.2f}      {f1_f.mean():.4f} +- {f1_f.std(ddof=1):.4f}")
    # Farklari "dogru siniflandirilan ornek sayisi" cinsinden tam sayiya cevir (kayan nokta es degerlerini bozmasin)
    n_test = int(round(0.2 * len(load_breast_cancer().target)))
    fark = np.rint((acc_m - acc_f) * n_test).astype(int)
    print(f"MLP onde: {(fark > 0).sum()}, FS-ANN onde: {(fark < 0).sum()}, esit: {(fark == 0).sum()}")
    print(f"Wilcoxon (dogruluk, MLP - FS-ANN): p = {wilcoxon(fark).pvalue:.3f}")


if __name__ == "__main__":
    main()
